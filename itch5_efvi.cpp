/**
 * itch5_efvi.cpp
 *
 * Zero-copy ITCH 5.0 receiver using Solarflare ef_vi + AVX2 message parsing.
 *
 * Build:
 *   g++ -O3 -march=native -std=c++20 -mavx2 \
 *       -I/usr/include/etherfabric \
 *       itch5_efvi.cpp -o itch5_efvi \
 *       -letherfabric -lpthread
 *
 * Run (as root or with CAP_NET_ADMIN):
 *   ./itch5_efvi eth1 239.192.0.1 26000
 */

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <atomic>
#include <array>
#include <string_view>
#include <bit>

#include <immintrin.h>          // AVX2
#include <arpa/inet.h>
#include <etherfabric/vi.h>
#include <etherfabric/pd.h>
#include <etherfabric/memreg.h>

// ─── Constants ───────────────────────────────────────────────────────────────

static constexpr int    RX_RING_SIZE   = 512;   // must be power-of-two
static constexpr int    PKT_BUF_SIZE   = 2048;  // per-buffer bytes
static constexpr int    N_BUFS         = 1024;
static constexpr size_t RING_CAPACITY  = 1u << 14; // 16384 slots

// ─── ITCH 5.0 message types (selected) ───────────────────────────────────────

enum class MsgType : uint8_t {
    SystemEvent     = 'S',
    StockDirectory  = 'R',
    AddOrder        = 'A',
    AddOrderMPID    = 'F',
    ExecuteOrder    = 'E',
    CancelOrder     = 'X',
    DeleteOrder     = 'D',
    ReplaceOrder    = 'U',
    Trade           = 'P',
};

#pragma pack(push, 1)

struct MoldUDP64Header {
    char     session[10];
    uint64_t seqno;        // big-endian
    uint16_t msg_count;    // big-endian
};

struct ITCHMsgHeader {
    uint16_t length;       // big-endian, excludes this field
    uint8_t  type;
};

struct AddOrderMsg {
    uint8_t  type;         // 'A'
    uint16_t stock_locate;
    uint16_t tracking_num;
    uint8_t  timestamp[6];
    uint64_t order_ref;
    char     side;         // 'B' or 'S'
    uint32_t shares;
    char     stock[8];
    uint32_t price;        // fixed-point 4 decimals
};

struct ExecuteOrderMsg {
    uint8_t  type;         // 'E'
    uint16_t stock_locate;
    uint16_t tracking_num;
    uint8_t  timestamp[6];
    uint64_t order_ref;
    uint32_t executed_shares;
    uint64_t match_num;
};

struct DeleteOrderMsg {
    uint8_t  type;         // 'D'
    uint16_t stock_locate;
    uint16_t tracking_num;
    uint8_t  timestamp[6];
    uint64_t order_ref;
};

#pragma pack(pop)

// ─── Lock-free SPSC ring (single producer, single consumer) ──────────────────

template<typename T, size_t N>
struct alignas(64) SPSCRing {
    static_assert((N & (N-1)) == 0, "N must be power of two");

    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    std::array<T, N> buf_;

    bool push(const T& v) noexcept {
        size_t t = tail_.load(std::memory_order_relaxed);
        size_t next = (t + 1) & (N - 1);
        if (next == head_.load(std::memory_order_acquire))
            return false;   // full
        buf_[t] = v;
        tail_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& out) noexcept {
        size_t h = head_.load(std::memory_order_relaxed);
        if (h == tail_.load(std::memory_order_acquire))
            return false;   // empty
        out = buf_[h];
        head_.store((h + 1) & (N - 1), std::memory_order_release);
        return true;
    }
};

// ─── Packet descriptor (what we put in the ring) ─────────────────────────────

struct PktDesc {
    uint8_t* data;
    uint32_t len;
    int      buf_id;    // ef_vi refill handle
};

static SPSCRing<PktDesc, RING_CAPACITY> g_ring;

// ─── ef_vi state ─────────────────────────────────────────────────────────────

struct EfviState {
    ef_driver_handle dh;
    ef_pd            pd;
    ef_vi            vi;
    ef_memreg        mr;

    uint8_t*  pkt_mem;      // huge aligned slab
    size_t    pkt_mem_sz;

    // per-buffer metadata
    struct BufMeta { ef_addr dma_addr; };
    std::array<BufMeta, N_BUFS> bufs;

    uint8_t* buf_ptr(int id) const {
        return pkt_mem + (size_t)id * PKT_BUF_SIZE;
    }
};

static EfviState g_ef;

// ─── AVX2 decoding ───────────────────────────────────────────────────────────
//
// Strategy: for each message type, issue a single VMOVDQU (32-byte load) then
// use VPSHUFB (_mm256_shuffle_epi8) to simultaneously:
//   (a) byte-swap every multi-byte BE field to host-endian LE, and
//   (b) pack the extracted fields into predictable register lanes,
// in one instruction.  Scalar _mm256_extract_epi{16,32,64} then pull clean
// host-endian values with no branches and no loop overhead.
//
// Fields that cross the 16-byte lane boundary (VPSHUFB cannot cross lanes) or
// fall outside the 32-byte window are handled by dedicated xmm primitives
// (VMOVQ + VPSHUFB on xmm, 2-cycle latency, fully pipelined).
//
// _mm256_set_epi8(e31,e30,...,e1,e0) argument convention:
//   arg at position P from the LEFT controls dst byte (31-P).
//   Positions 0-15  → dst bytes 31-16 (lane 1, local indices 15-0)
//   Positions 16-31 → dst bytes 15- 0 (lane 0, local indices 15-0)
//   Each value is a LOCAL lane byte index (0-15); -1 zeros that dst byte.
//
// shuffle arg ordering for a bswap of a BE u16 at lane0 offsets [lo_idx, hi_idx]
// into dst bytes [d_lo, d_lo+1]:
//   dst[d_lo]   = lane0[hi_idx]   ← MSB of BE  → LSB of LE
//   dst[d_lo+1] = lane0[lo_idx]   ← LSB of BE  → MSB of LE
//   In set_epi8: pos (31 - d_lo)   = hi_idx
//                pos (31 - d_lo-1) = lo_idx
//
// All shuffle maps below are derived analytically and verified by unit tests
// (see companion itch5_avx_test.cpp).  The tests cover all three message types
// with known wire vectors and assert every decoded field.

// ── Unaligned load helpers ────────────────────────────────────────────────────
//
// ITCH messages are packed back-to-back inside MoldUDP64 datagrams.  After
// the first message the cursor advances by (2 + mlen) bytes, so every
// subsequent message body starts at an ARBITRARY byte offset — there is no
// alignment guarantee at all.
//
// C++ aliasing rules: dereferencing a T* that is not suitably aligned for T
// is undefined behaviour.  On x86 unaligned integer reads happen to work at
// runtime, but the compiler is free to assume the pointer IS aligned and emit
// instructions (e.g. MOVAPS) that fault on unaligned addresses, or reorder /
// eliminate loads based on the false alignment assumption.
//
// Safe pattern: use memcpy (elided to a single MOV by every modern compiler)
// or the Intel-blessed "loadu" intrinsics that carry the unaligned contract.
//
// Intrinsic alignment contracts (Intel Intrinsics Guide):
//   _mm256_loadu_si256  — no alignment requirement  ✓
//   _mm_loadu_si64      — no alignment requirement  ✓  (GCC/Clang: SSE2)
//   _mm_loadu_si32      — no alignment requirement  ✓  (GCC 11+ / Clang 10+)
//   _mm_loadl_epi64(p)  — p must be 8-byte aligned  ✗  (despite the name)
//   _mm_cvtsi32_si128(*reinterpret_cast<uint32_t*>(p)) — UB if p unaligned  ✗
//
// We replace the two broken primitives with __builtin_memcpy-into-scalar,
// which the compiler lowers to a single unaligned integer load (MOVQ / MOV r32).

// bswap64_xmm: load 8 unaligned bytes → reverse byte order → host-endian u64.
// Emits: MOVQ xmm + VPSHUFB xmm — 2-cycle latency, no alignment fault risk.
static inline uint64_t bswap64_xmm(const uint8_t* __restrict__ p) noexcept {
    // __builtin_memcpy → compiler emits MOVQ xmm (zero-extends), no UB.
    uint64_t raw;
    __builtin_memcpy(&raw, p, 8);
    __m128i v = _mm_cvtsi64_si128(static_cast<long long>(raw));
    // Shuffle: dst[i] = src[7-i]  — byte-reversal inside the low 8 bytes.
    // _mm_set_epi8(e15..e0): e0=dst[0]=src[7], e1=dst[1]=src[6], ..., e7=dst[7]=src[0].
    const __m128i sh = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1, 0,1,2,3,4,5,6,7);
    return static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_shuffle_epi8(v, sh)));
}

// bswap32_xmm: load 4 unaligned bytes → reverse → host-endian u32.
// Emits: MOV r32 (unaligned scalar) + VMOVD xmm + VPSHUFB xmm.
static inline uint32_t bswap32_xmm(const uint8_t* __restrict__ p) noexcept {
    uint32_t raw;
    __builtin_memcpy(&raw, p, 4);                 // single MOV r32, no UB
    __m128i v = _mm_cvtsi32_si128(static_cast<int>(raw));
    // Shuffle: dst[i] = src[3-i]  — byte-reversal inside the low 4 bytes.
    // e3=dst[3]=src[0], e2=dst[2]=src[1], e1=dst[1]=src[2], e0=dst[0]=src[3].
    const __m128i sh = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0,1,2,3);
    return static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_shuffle_epi8(v, sh)));
}

// ymm_loadu: unaligned 32-byte load with explicit aliasing safety.
// The standard cast-to-__m256i* is technically UB for unaligned pointers under
// strict-aliasing rules.  __m256i_u (GCC/Clang <immintrin.h>) carries
// __attribute__((__may_alias__, __aligned__(1))), so the compiler knows the
// load is intentionally unaligned and will not propagate alignment assumptions.
static inline __m256i ymm_loadu(const uint8_t* __restrict__ p) noexcept {
    return *reinterpret_cast<const __m256i_u*>(p);
}

// ── AddOrder ('A') / AddOrderMPID ('F') ─────────────────────────────────────
//
// Wire body layout (36 bytes):
//   [0]     type
//   [1-2]   stock_locate  BE u16
//   [3-4]   tracking_num  BE u16
//   [5-10]  timestamp     BE u48  (nanoseconds past midnight)
//   [11-18] order_ref     BE u64
//   [19]    side          'B' or 'S'
//   [20-23] shares        BE u32
//   [24-31] stock         ASCII, space-padded
//   [32-35] price         BE u32  (fixed-point ×10000)
//
// One 32-byte VMOVDQU covers bytes [0-31] (both lanes).
// Lane 0 = msg[0-15]:  extracts stock_locate, tracking_num
// Lane 1 = msg[16-31]: extracts shares (bswap), stock (ASCII, no swap)
// order_ref (msg[11-18]) straddles the lane boundary → bswap64_xmm.
// timestamp (msg[5-10], 6 bytes) → bswap64_xmm(msg+5) >> 16 (drops 2 low bytes).
// price (msg[32-35]) is outside the 32-byte window → bswap32_xmm.
//
// Shuffle map (verified):
//   dst[ 0- 1] = bswap(lane0[1-2])  → stock_locate LE
//   dst[ 2- 3] = bswap(lane0[3-4])  → tracking_num LE
//   dst[16-19] = bswap(lane1[4-7])  → shares LE
//   dst[24-31] = lane1[8-15]        → stock[0-7] (no swap, ASCII)
//   all other dst bytes = 0

struct DecodedAddOrder {
    uint64_t order_ref;
    uint32_t shares;
    uint32_t price;
    uint16_t stock_locate;
    uint16_t tracking_num;
    char     stock[9];      // NUL-terminated
    char     side;
    uint64_t timestamp_ns;  // nanoseconds past midnight, u48 value in u64
};

static inline DecodedAddOrder avx_decode_add_order(const uint8_t* msg) noexcept {
    __m256i raw = ymm_loadu(msg);

    // Shuffle derivation (positions = arg index from LEFT in set_epi8 call):
    //
    // stock_locate: dst[0]=lane0[2], dst[1]=lane0[1]
    //   pos31 (→ dst[0]) = 2,  pos30 (→ dst[1]) = 1
    // tracking_num: dst[2]=lane0[4], dst[3]=lane0[3]
    //   pos29 (→ dst[2]) = 4,  pos28 (→ dst[3]) = 3
    // shares (bswap): dst[16]=lane1[7], dst[17]=lane1[6], dst[18]=lane1[5], dst[19]=lane1[4]
    //   pos15 (→ dst[16]) = 7, pos14 (→ dst[17]) = 6,
    //   pos13 (→ dst[18]) = 5, pos12 (→ dst[19]) = 4
    //   → args at positions 12-15 in LEFT-to-RIGHT order = 4, 5, 6, 7
    // stock (no swap): dst[24]=lane1[8], ..., dst[31]=lane1[15]
    //   pos7 (→ dst[24]) = 8, pos6 (→ dst[25]) = 9, ..., pos0 (→ dst[31]) = 15
    //   → args at positions 0-7 = 15,14,13,12,11,10,9,8
    const __m256i shuf = _mm256_set_epi8(
        // pos  0- 7 → dst[31-24] : stock[7..0] = lane1[15..8]
        15,14,13,12,11,10, 9, 8,
        // pos  8-11 → dst[23-20] : zeros
        -1,-1,-1,-1,
        // pos 12-15 → dst[19-16] : shares bswap = lane1[4..7] reversed
         4, 5, 6, 7,
        // pos 16-27 → dst[15- 4] : zeros
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        // pos 28-31 → dst[ 3- 0] : tracking_num, stock_locate (both bswapped)
         3, 4, 1, 2
    );
    __m256i out = _mm256_shuffle_epi8(raw, shuf);

    // extract_epi16(n) → 16-bit word n (byte offset 2n in the register)
    uint16_t stock_locate = static_cast<uint16_t>(_mm256_extract_epi16(out, 0));
    uint16_t tracking_num = static_cast<uint16_t>(_mm256_extract_epi16(out, 1));
    // extract_epi32(4) → 32-bit word 4 = bytes 16-19
    uint32_t shares       = static_cast<uint32_t>(_mm256_extract_epi32(out, 4));
    // extract_epi64(3) → 64-bit word 3 = bytes 24-31 (stock ASCII)
    uint64_t stock_lo     = static_cast<uint64_t>(_mm256_extract_epi64(out, 3));

    // Cross-lane and out-of-window fields via xmm primitives:
    uint64_t ts_raw       = bswap64_xmm(msg + 5);
    uint64_t timestamp_ns = ts_raw >> 16;   // u48 in top 6 bytes after bswap
    uint64_t order_ref    = bswap64_xmm(msg + 11);
    char     side         = static_cast<char>(msg[19]);
    uint32_t price        = bswap32_xmm(msg + 32);

    DecodedAddOrder r{};
    r.stock_locate = stock_locate;
    r.tracking_num = tracking_num;
    r.order_ref    = order_ref;
    r.side         = side;
    r.shares       = shares;
    r.price        = price;
    r.timestamp_ns = timestamp_ns;
    memcpy(r.stock, &stock_lo, 8);
    r.stock[8] = '\0';
    return r;
}

// ── ExecuteOrder ('E') ───────────────────────────────────────────────────────
//
// Wire body layout (31 bytes):
//   [0]     type
//   [1-2]   stock_locate    BE u16
//   [3-4]   tracking_num    BE u16
//   [5-10]  timestamp       BE u48
//   [11-18] order_ref       BE u64
//   [19-22] executed_shares BE u32  → lane1 local offsets [3-6]
//   [23-30] match_num       BE u64
//
// 32-byte load covers all 31 bytes (1 byte pad, buffer always ≥ 2 KB).
// Lane 1 = msg[16-31]: executed_shares at local offsets [3-6].
// executed_shares bswap: dst[16]=lane1[6], dst[17]=lane1[5], dst[18]=lane1[4], dst[19]=lane1[3]
//   → args at positions 12-15 = 3, 4, 5, 6

struct DecodedExecuteOrder {
    uint64_t order_ref;
    uint64_t match_num;
    uint32_t executed_shares;
    uint16_t stock_locate;
    uint16_t tracking_num;
    uint64_t timestamp_ns;
};

static inline DecodedExecuteOrder avx_decode_execute_order(const uint8_t* msg) noexcept {
    __m256i raw = ymm_loadu(msg);

    const __m256i shuf = _mm256_set_epi8(
        // pos  0-11 → dst[31-20] : zeros
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        // pos 12-15 → dst[19-16] : executed_shares bswap (lane1[3-6] reversed)
         3, 4, 5, 6,
        // pos 16-27 → dst[15- 4] : zeros
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        // pos 28-31 → dst[ 3- 0] : tracking_num, stock_locate (bswapped)
         3, 4, 1, 2
    );
    __m256i out = _mm256_shuffle_epi8(raw, shuf);

    uint16_t stock_locate    = static_cast<uint16_t>(_mm256_extract_epi16(out, 0));
    uint16_t tracking_num    = static_cast<uint16_t>(_mm256_extract_epi16(out, 1));
    uint32_t executed_shares = static_cast<uint32_t>(_mm256_extract_epi32(out, 4));

    uint64_t ts_raw       = bswap64_xmm(msg + 5);
    uint64_t timestamp_ns = ts_raw >> 16;
    uint64_t order_ref    = bswap64_xmm(msg + 11);
    uint64_t match_num    = bswap64_xmm(msg + 23);

    return { order_ref, match_num, executed_shares, stock_locate, tracking_num, timestamp_ns };
}

// ── DeleteOrder ('D') ────────────────────────────────────────────────────────
//
// Wire body layout (19 bytes):
//   [0]     type
//   [1-2]   stock_locate  BE u16
//   [3-4]   tracking_num  BE u16
//   [5-10]  timestamp     BE u48
//   [11-18] order_ref     BE u64
//
// 19 bytes < 32; 32-byte load is safe (ef_vi buffers are PKT_BUF_SIZE = 2 KB).

struct DecodedDeleteOrder {
    uint64_t order_ref;
    uint16_t stock_locate;
    uint16_t tracking_num;
    uint64_t timestamp_ns;
};

static inline DecodedDeleteOrder avx_decode_delete_order(const uint8_t* msg) noexcept {
    __m256i raw = ymm_loadu(msg);

    const __m256i shuf = _mm256_set_epi8(
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,   // lane1 unused
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,                // dst[15-4] unused
         3, 4, 1, 2                                          // tracking, stock_locate
    );
    __m256i out = _mm256_shuffle_epi8(raw, shuf);

    uint16_t stock_locate = static_cast<uint16_t>(_mm256_extract_epi16(out, 0));
    uint16_t tracking_num = static_cast<uint16_t>(_mm256_extract_epi16(out, 1));

    uint64_t ts_raw       = bswap64_xmm(msg + 5);
    uint64_t timestamp_ns = ts_raw >> 16;
    uint64_t order_ref    = bswap64_xmm(msg + 11);

    return { order_ref, stock_locate, tracking_num, timestamp_ns };
}

// ── Type scanner: find interesting message type bytes in a 32-byte window ────
// Returns bitmask: bit i set ↔ byte i is one of A/F/E/D/U.
static inline uint32_t avx_scan_types(const uint8_t* p) noexcept {
    __m256i data = ymm_loadu(p);
    __m256i any  = _mm256_or_si256(
        _mm256_or_si256(_mm256_cmpeq_epi8(data, _mm256_set1_epi8('A')),
                        _mm256_cmpeq_epi8(data, _mm256_set1_epi8('F'))),
        _mm256_or_si256(
            _mm256_or_si256(_mm256_cmpeq_epi8(data, _mm256_set1_epi8('E')),
                            _mm256_cmpeq_epi8(data, _mm256_set1_epi8('D'))),
            _mm256_cmpeq_epi8(data, _mm256_set1_epi8('U'))));
    return static_cast<uint32_t>(_mm256_movemask_epi8(any));
}

// ─── ITCH message dispatcher ─────────────────────────────────────────────────

static void on_add_order(const DecodedAddOrder& m) {
    double price = m.price / 10000.0;
    printf("[ADD] %-8s %c %6u @ %9.4f  ref=%-20lu  ts=%lu ns\n",
           m.stock, m.side, m.shares, price,
           (unsigned long)m.order_ref, (unsigned long)m.timestamp_ns);
}

static void on_execute(const DecodedExecuteOrder& m) {
    printf("[EXE] ref=%-20lu  qty=%6u  match=%-20lu  ts=%lu ns\n",
           (unsigned long)m.order_ref, m.executed_shares,
           (unsigned long)m.match_num, (unsigned long)m.timestamp_ns);
}

static void on_delete(const DecodedDeleteOrder& m) {
    printf("[DEL] ref=%-20lu  ts=%lu ns\n",
           (unsigned long)m.order_ref, (unsigned long)m.timestamp_ns);
}

// ─── Parse a single MoldUDP64 datagram ───────────────────────────────────────

static void parse_datagram(const uint8_t* pkt, uint32_t len) noexcept {
    // pkt points to the MoldUDP64 UDP payload at an arbitrary byte offset
    // inside a 2 KB ef_vi DMA buffer — no alignment guarantee.
    //
    // MoldUDP64 header layout (22 bytes):
    //   [0-9]   session   char[10]
    //   [10-17] seqno     BE u64
    //   [18-19] msg_count BE u16
    //
    // Do NOT cast pkt to MoldUDP64Header* and dereference multi-byte fields.
    // Even with __attribute__((packed)) the base pointer may be unaligned, and
    // dereferencing a packed struct member via an unaligned pointer is still UB
    // once the compiler can prove alignment.  Use __builtin_memcpy instead.

    if (len < 22u) return;

    uint16_t msg_count_be;
    __builtin_memcpy(&msg_count_be, pkt + 18, 2);
    uint16_t msg_count = __builtin_bswap16(msg_count_be);

    if (msg_count == 0) return;

    const uint8_t* cur = pkt + 22;   // first MoldUDP message block
    const uint8_t* end = pkt + len;

    while (msg_count-- && cur + 3 <= end) {
        // Per-message block:
        //   [0-1]  length  BE u16  (body length, not including these 2 bytes)
        //   [2]    type    uint8
        //   [3..]  fields
        uint16_t mlen_be;
        __builtin_memcpy(&mlen_be, cur, 2);
        uint16_t mlen = __builtin_bswap16(mlen_be);

        if (mlen < 1 || cur + 2 + mlen > end) break;

        const uint8_t  type = cur[2];   // single byte: no alignment issue
        const uint8_t* body = cur + 2;  // body[0] = type

        // ── Prefetch next message into L1 while decoding this one ─────────────
        // The decode path has several dependent loads (ymm + xmm).  Issuing a
        // T0 prefetch for the next message's first cache line now hides DRAM
        // latency behind the current message's AVX shuffle work.
        const uint8_t* next = cur + 2 + mlen;
        if (next + 3 <= end)
            _mm_prefetch(reinterpret_cast<const char*>(next + 2), _MM_HINT_T0);

        // ── Fast type-filter via AVX ──────────────────────────────────────────
        // avx_scan_types loads 32 bytes from body.  The load may bleed into the
        // next message's length prefix, but we only inspect bit 0 of the result
        // mask (= body[0] = type byte), so no cross-message false positives.
        // The 32-byte overread is safe: ef_vi buffers are PKT_BUF_SIZE (2 KB),
        // so any byte within the buffer has at least 32 bytes of valid mapped
        // memory after it.
        {
            uint32_t mask = avx_scan_types(body);
            if (!(mask & 1u)) {   // bit 0 = body[0] = type
                cur = next;
                continue;
            }
        }

        switch (static_cast<MsgType>(type)) {
            case MsgType::AddOrder:
            case MsgType::AddOrderMPID:
                if (mlen >= 36) on_add_order(avx_decode_add_order(body));
                break;
            case MsgType::ExecuteOrder:
                if (mlen >= 31) on_execute(avx_decode_execute_order(body));
                break;
            case MsgType::DeleteOrder:
                if (mlen >= 19) on_delete(avx_decode_delete_order(body));
                break;
            default:
                break;
        }
        cur = next;
    }
}

// ─── Consumer thread ─────────────────────────────────────────────────────────

static void* consumer_thread(void*) {
    PktDesc desc;
    while (true) {
        if (g_ring.pop(desc)) {
            // UDP payload starts after Ethernet(14) + IP(20) + UDP(8) = 42 bytes
            constexpr int HDR = 42;
            if (desc.len > HDR)
                parse_datagram(desc.data + HDR, desc.len - HDR);

            // Return buffer to ef_vi RX ring
            ef_vi_receive_init(&g_ef.vi,
                               g_ef.bufs[desc.buf_id].dma_addr,
                               desc.buf_id);
            ef_vi_receive_push(&g_ef.vi);
        }
    }
    return nullptr;
}

// ─── ef_vi initialisation ────────────────────────────────────────────────────

static void efvi_init(const char* iface) {
    int rc;

    rc = ef_driver_open(&g_ef.dh);
    if (rc < 0) { perror("ef_driver_open"); exit(1); }

    rc = ef_pd_alloc_by_name(&g_ef.pd, g_ef.dh, iface, EF_PD_DEFAULT);
    if (rc < 0) { perror("ef_pd_alloc_by_name"); exit(1); }

    rc = ef_vi_alloc_from_pd(&g_ef.vi, g_ef.dh, &g_ef.pd, g_ef.dh,
                              RX_RING_SIZE, 0, -1, nullptr, -1,
                              EF_VI_FLAGS_DEFAULT);
    if (rc < 0) { perror("ef_vi_alloc_from_pd"); exit(1); }

    // Allocate packet memory (huge-page aligned).
    g_ef.pkt_mem_sz = (size_t)N_BUFS * PKT_BUF_SIZE;
    rc = posix_memalign(reinterpret_cast<void**>(&g_ef.pkt_mem),
                        4096, g_ef.pkt_mem_sz);
    if (rc) { perror("posix_memalign"); exit(1); }

    rc = ef_memreg_alloc(&g_ef.mr, g_ef.dh, &g_ef.pd, g_ef.dh,
                         g_ef.pkt_mem, g_ef.pkt_mem_sz);
    if (rc < 0) { perror("ef_memreg_alloc"); exit(1); }

    // Map DMA addresses and post initial RX buffers.
    for (int i = 0; i < N_BUFS; ++i) {
        g_ef.bufs[i].dma_addr =
            ef_memreg_dma_addr(&g_ef.mr, (size_t)i * PKT_BUF_SIZE);
    }
    for (int i = 0; i < RX_RING_SIZE; ++i) {
        ef_vi_receive_init(&g_ef.vi, g_ef.bufs[i].dma_addr, i);
    }
    ef_vi_receive_push(&g_ef.vi);

    printf("[efvi] interface=%s vi_rxq_size=%d\n",
           iface, ef_vi_receive_capacity(&g_ef.vi));
}

static void efvi_add_mcast_filter(const char* group_ip, uint16_t port) {
    ef_filter_spec fs;
    ef_filter_spec_init(&fs, EF_FILTER_FLAG_NONE);

    struct sockaddr_in sin{};
    sin.sin_family = AF_INET;
    inet_pton(AF_INET, group_ip, &sin.sin_addr);
    sin.sin_port = htons(port);

    int rc = ef_filter_spec_set_ip4_local(&fs, IPPROTO_UDP,
                                           sin.sin_addr.s_addr,
                                           sin.sin_port);
    if (rc < 0) { perror("ef_filter_spec_set_ip4_local"); exit(1); }

    ef_filter_cookie cookie;
    rc = ef_vi_filter_add(&g_ef.vi, g_ef.dh, &fs, &cookie);
    if (rc < 0) { perror("ef_vi_filter_add"); exit(1); }

    printf("[efvi] filter: udp %s:%u\n", group_ip, port);
}

// ─── Main polling loop ───────────────────────────────────────────────────────

static void poll_loop() {
    static ef_event evts[64];

    while (true) {
        int n = ef_eventq_poll(&g_ef.vi, evts, 64);
        if (n == 0) [[likely]] {
            _mm_pause();
            continue;
        }

        for (int i = 0; i < n; ++i) {
            if (EF_EVENT_TYPE(evts[i]) != EF_EVENT_TYPE_RX) continue;

            int   id  = EF_EVENT_RX_RQ_ID(evts[i]);
            uint32_t len = EF_EVENT_RX_BYTES(evts[i]);
            uint8_t* data = g_ef.buf_ptr(id);

            PktDesc desc{ data, len, id };
            while (!g_ring.push(desc)) _mm_pause();  // back-pressure spin
        }
    }
}

// ─── Entry point ─────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <iface> <mcast-group> <port>\n", argv[0]);
        return 1;
    }

    const char*    iface    = argv[1];
    const char*    mcast_ip = argv[2];
    uint16_t       port     = static_cast<uint16_t>(atoi(argv[3]));

    efvi_init(iface);
    efvi_add_mcast_filter(mcast_ip, port);

    pthread_t tid;
    pthread_create(&tid, nullptr, consumer_thread, nullptr);
    pthread_detach(tid);

    printf("[main] entering poll loop\n");
    poll_loop();   // never returns
}
