// efvi_md.hpp — ef_vi receiver for HFT market data: multiple multicast
// channels x multiple sides (A/B lines), single poll thread, zero-copy.
//
// Shape
//   Port     = one ef_vi on one NIC port. A and B lines arrive on physically
//              separate ports, so SIDE is a property of the port, never parsed.
//              All channels on that port share the VI (one multicast filter
//              each); demux is by (dst IP, dst port).
//   Channel  = one feed (e.g. ITCH group / UQDF group) with one MoldUDP64
//              sequence space. Both sides feed the same per-channel Arbiter:
//              first arrival wins by sequence, duplicates dropped, gaps counted.
//   Receiver = N ports + arbiters + a user Sink invoked inline on the poll
//              thread for every accepted message (zero-copy: pointer into the
//              DMA buffer; consume before returning).
//
// Fast path per packet
//   1. event batch from ef_eventq_poll; while handling event i, prefetch the
//      buffer of event i+1 (header line + payload line) — the only two lines
//      the fast path touches
//   2. AVX-512BW: one masked 64-byte compare validates Ethernet II / IPv4 /
//      UDP fixed fields in a single instruction
//   3. AVX-512F: (dst ip, dst port) key against the port's channel table,
//      8 channels per compare, tzcnt on the mask
//   4. MoldUDP64 header (session 10, seq 8 BE, count 2 BE) -> arbiter ->
//      Sink::on_msg(chan, seq, msg, len, ts, side) per message
//   5. descriptor returned to the free stack; refill in batches with ONE
//      doorbell (ef_vi_receive_push) per batch
// Fallbacks: AVX2 / scalar for the classify + lookup steps.
//
// Limits (stated, counted, never silent): jumbo frames need rx_buf_size >=
// 9216 + prefix (set 16384 for Nasdaq's FPGA ITCH feed); packets that span
// descriptors (SOP/CONT) are dropped and counted, not reassembled.
//
// Build (real):  g++ -std=c++23 -O3 -march=emeraldrapids app.cpp -lciul1
// Build (mock):  g++ -std=c++23 -O3 -march=emeraldrapids -Imock ...

#pragma once

#include <etherfabric/ef_vi.h>
#include <etherfabric/vi.h>
#include <etherfabric/pd.h>
#include <etherfabric/memreg.h>

#include <bit>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/mman.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#if defined(__AVX512F__) && defined(__AVX512BW__)
#  include <immintrin.h>
#  define EFVI_MD_AVX512 1
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define EFVI_MD_AVX2 1
#endif

namespace md::net {

// ---------------------------------------------------------------- wire
namespace wire {
constexpr uint32_t kEthTypeOff = 12, kIpVerOff = 14, kIpTotLenOff = 16,
                   kIpProtoOff = 23, kIpDstOff = 30, kUdpDstOff = 36,
                   kUdpLenOff = 38, kUdpPayloadOff = 42;
constexpr uint32_t kMoldHdr = 20;     // session[10] seq[8] count[2]

inline uint16_t be16(const uint8_t* p) noexcept {
    uint16_t v; std::memcpy(&v, p, 2); return std::byteswap(v); }
inline uint32_t be32(const uint8_t* p) noexcept {
    uint32_t v; std::memcpy(&v, p, 4); return std::byteswap(v); }
inline uint64_t be64(const uint8_t* p) noexcept {
    uint64_t v; std::memcpy(&v, p, 8); return std::byteswap(v); }
}  // namespace wire

// ---------------------------------------------------------------- config
struct Config {
    uint32_t rx_buf_size  = 4096;  // per descriptor; >= 9216+prefix for jumbo
    uint32_t rx_ring      = 1024;  // descriptors kept posted
    uint32_t evq_batch    = 64;    // events per ef_eventq_poll
    uint32_t refill_batch = 32;    // descriptors per doorbell
    bool     hw_timestamps = true;
    bool     event_merge   = true; // EF_VI_RX_EVENT_MERGE (RX_MULTI events)
};

struct Stats {
    uint64_t pkts = 0, msgs = 0, bytes = 0;
    uint64_t dups = 0, gaps = 0, gap_msgs = 0, heartbeats = 0;
    uint64_t bad_hdr = 0, unknown_dst = 0, discards = 0, spans = 0,
             no_desc = 0, short_pkt = 0;
};

// ---------------------------------------------------------------- channel table
// key = (dst_ip << 16) | dst_port, 0 = empty. Up to 64 channels per port.
struct alignas(64) ChannelTable {
    static constexpr unsigned kMax = 64;
    uint64_t key[kMax]{};
    uint8_t  chan[kMax]{};
    uint32_t n = 0;

    bool add(uint32_t ip, uint16_t port, uint8_t ch) noexcept {
        if (n == kMax) return false;
        key[n] = (uint64_t(ip) << 16) | port; chan[n] = ch; ++n; return true;
    }
    // -1 if absent
    [[nodiscard]] int lookup(uint64_t k) const noexcept {
#if defined(EFVI_MD_AVX512)
        const __m512i needle = _mm512_set1_epi64(static_cast<long long>(k));
        for (unsigned i = 0; i < n; i += 8) {
            const __m512i v = _mm512_load_si512(key + i);
            const __mmask8 m = _mm512_cmpeq_epi64_mask(v, needle);
            if (m) return chan[i + std::countr_zero(static_cast<unsigned>(m))];
        }
        return -1;
#elif defined(EFVI_MD_AVX2)
        const __m256i needle = _mm256_set1_epi64x(static_cast<long long>(k));
        for (unsigned i = 0; i < n; i += 4) {
            const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(key + i));
            const int m = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v, needle)));
            if (m) return chan[i + std::countr_zero(static_cast<unsigned>(m))];
        }
        return -1;
#else
        for (unsigned i = 0; i < n; ++i) if (key[i] == k) return chan[i];
        return -1;
#endif
    }
};

// ---------------------------------------------------------------- header classify
// Ethernet II (0x0800) / IPv4 no options (0x45) / UDP (17). One masked
// byte-compare over the first 64 bytes on AVX-512BW. Buffer is always >= 64
// readable bytes (descriptor size), so the load is safe; only bytes < len
// are trusted, and the mask selects fixed-position bytes inside 42.
struct HeaderClassifier {
#if defined(EFVI_MD_AVX512)
    __m512i tmpl;
    __mmask64 mask;
    HeaderClassifier() noexcept {
        alignas(64) uint8_t t[64]{};
        t[12] = 0x08; t[13] = 0x00; t[14] = 0x45; t[23] = 17;
        tmpl = _mm512_load_si512(t);
        mask = (1ull << 12) | (1ull << 13) | (1ull << 14) | (1ull << 23);
    }
    [[nodiscard]] bool ok(const uint8_t* f) const noexcept {
        const __m512i v = _mm512_loadu_si512(f);
        return _mm512_mask_cmpeq_epi8_mask(mask, v, tmpl) == mask;
    }
#else
    [[nodiscard]] bool ok(const uint8_t* f) const noexcept {
        return f[12] == 0x08 && f[13] == 0x00 && f[14] == 0x45 && f[23] == 17;
    }
#endif
};

// ---------------------------------------------------------------- arbiter
// MoldUDP64 line arbitration: both sides feed it; deliver seq >= next in
// arrival order, drop the rest. No reorder buffer (that is a policy choice:
// lowest latency; a late lower seq from the slow side is simply a dup).
class Arbiter {
public:
    template <class Sink>
    void on_packet(uint8_t chan, uint8_t side, const uint8_t* payload,
                   uint32_t len, uint64_t ts_ns, Sink& sink, Stats& st) noexcept {
        if (len < wire::kMoldHdr) [[unlikely]] { ++st.short_pkt; return; }
        const uint64_t seq   = wire::be64(payload + 10);
        const uint32_t count = wire::be16(payload + 18);
        if (count == 0 || count == 0xFFFF) {           // heartbeat / end of session
            ++st.heartbeats;
            if (synced_ && seq > next_) { ++st.gaps; st.gap_msgs += seq - next_; next_ = seq; }
            return;
        }
        if (!synced_) { next_ = seq; synced_ = true; }
        if (seq + count <= next_) { ++st.dups; return; }    // whole packet stale
        if (seq > next_) { ++st.gaps; st.gap_msgs += seq - next_; next_ = seq; }

        const uint8_t* p = payload + wire::kMoldHdr;
        const uint8_t* end = payload + len;
        uint64_t s = seq;
        for (uint32_t i = 0; i < count; ++i, ++s) {
            if (p + 2 > end) [[unlikely]] { ++st.short_pkt; break; }
            const uint32_t mlen = wire::be16(p); p += 2;
            if (p + mlen > end) [[unlikely]] { ++st.short_pkt; break; }
            if (s >= next_) {                                // skip stale prefix
                sink.on_msg(chan, side, s, p, mlen, ts_ns);
                ++st.msgs;
                next_ = s + 1;
            }
            p += mlen;
        }
    }
    [[nodiscard]] uint64_t next_seq() const noexcept { return next_; }
private:
    uint64_t next_ = 0;
    bool synced_ = false;
};

// ---------------------------------------------------------------- port (one ef_vi)
class Port {
public:
    Port() = default;
    Port(const Port&) = delete;
    Port& operator=(const Port&) = delete;
    ~Port() { close(); }

    int open(int ifindex, uint8_t side, const Config& cfg) {
        cfg_ = cfg; side_ = side;
        if (ef_driver_open(&dh_) < 0) return -1;
        if (ef_pd_alloc(&pd_, dh_, ifindex, EF_PD_DEFAULT) < 0) return -2;
        unsigned flags = EF_VI_FLAGS_DEFAULT;
        if (cfg.hw_timestamps) flags |= EF_VI_RX_TIMESTAMPS;
        if (cfg.event_merge)   flags |= EF_VI_RX_EVENT_MERGE;
        if (ef_vi_alloc_from_pd(&vi_, dh_, &pd_, dh_, -1, static_cast<int>(cfg.rx_ring), 0,
                                nullptr, -1, static_cast<enum ef_vi_flags>(flags)) < 0) return -3;
        prefix_ = static_cast<uint32_t>(ef_vi_receive_prefix_len(&vi_));
        nbufs_ = cfg.rx_ring + cfg.refill_batch;      // slack so refill never starves
        buf_size_ = cfg.rx_buf_size;
        mem_size_ = static_cast<size_t>(nbufs_) * buf_size_;
        mem_ = alloc_dma_impl(mem_size_);
        if (!mem_) return -4;
        if (ef_memreg_alloc(&mr_, dh_, &pd_, dh_, mem_, mem_size_) < 0) return -5;
        free_ = static_cast<uint32_t*>(std::malloc(nbufs_ * sizeof(uint32_t)));
        nfree_ = 0;
        for (uint32_t i = nbufs_; i-- > 0;) free_[nfree_++] = i;
        refill(true);
        open_ = true;
        return 0;
    }
    void close() noexcept {
        if (!open_) return;
        ef_memreg_free(&mr_, dh_); ef_vi_free(&vi_, dh_); ef_pd_free(&pd_, dh_);
        ef_driver_close(dh_);
        if (mem_) { if (huge_) munmap(mem_, mem_size_); else std::free(mem_); }
        std::free(free_);
        open_ = false;
    }

    // multicast group for a channel on this port (side is the port's)
    int add_channel(uint8_t chan, uint32_t group_ip_host, uint16_t port_host) {
        if (!table_.add(group_ip_host, port_host, chan)) return -1;
        ef_filter_spec fs; ef_filter_spec_init(&fs, EF_FILTER_FLAG_NONE);
        ef_filter_spec_set_ip4_local(&fs, IPPROTO_UDP, htonl(group_ip_host), htons(port_host));
        ef_filter_cookie ck;
        return ef_vi_filter_add(&vi_, dh_, &fs, &ck);
    }

    ef_vi* vi() noexcept { return &vi_; }
    uint8_t side() const noexcept { return side_; }
    const Stats& stats() const noexcept { return st_; }

    // One poll: drains up to evq_batch events. Returns events handled.
    template <class Deliver>
    int poll(Deliver& deliver) noexcept {
        ef_event evs[256];
        const int n = ef_eventq_poll(&vi_, evs, static_cast<int>(cfg_.evq_batch));
        for (int i = 0; i < n; ++i) {
            if (i + 1 < n && EF_EVENT_TYPE(evs[i + 1]) == EF_EVENT_TYPE_RX) [[likely]]
                prefetch_pkt(EF_EVENT_RX_RQ_ID(evs[i + 1]));
            switch (EF_EVENT_TYPE(evs[i])) {
            case EF_EVENT_TYPE_RX: {
                const uint32_t id = EF_EVENT_RX_RQ_ID(evs[i]);
                if (EF_EVENT_RX_SOP(evs[i]) && !EF_EVENT_RX_CONT(evs[i])) [[likely]]
                    handle(id, EF_EVENT_RX_BYTES(evs[i]), deliver);
                else ++st_.spans;
                release(id);
                break;
            }
            case EF_EVENT_TYPE_RX_MULTI: {
                ef_request_id ids[64];
                const int k = ef_vi_receive_unbundle(&vi_, &evs[i], ids);
                const bool whole = EF_EVENT_RX_MULTI_SOP(evs[i]) && !EF_EVENT_RX_MULTI_CONT(evs[i]);
                for (int j = 0; j < k; ++j) {
                    if (j + 1 < k) prefetch_pkt(ids[j + 1]);
                    if (whole) [[likely]] {
                        uint16_t bytes = 0;
                        ef_vi_receive_get_bytes(&vi_, buf(ids[j]), &bytes);
                        handle(ids[j], bytes, deliver);
                    } else ++st_.spans;
                    release(ids[j]);
                }
                break;
            }
            case EF_EVENT_TYPE_RX_DISCARD:
                ++st_.discards; release(EF_EVENT_RX_RQ_ID(evs[i])); break;
            case EF_EVENT_TYPE_RX_MULTI_DISCARD: {
                ef_request_id ids[64];
                const int k = ef_vi_receive_unbundle(&vi_, &evs[i], ids);
                st_.discards += static_cast<uint64_t>(k);
                for (int j = 0; j < k; ++j) release(ids[j]);
                break;
            }
            case EF_EVENT_TYPE_RX_NO_DESC_TRUNC:
                ++st_.no_desc; break;
            default: break;
            }
        }
        if (nfree_ >= cfg_.refill_batch) refill(false);
        return n;
    }

private:
    uint8_t* buf(uint32_t id) const noexcept { return mem_ + static_cast<size_t>(id) * buf_size_; }
    void prefetch_pkt(uint32_t id) const noexcept {
        const uint8_t* p = buf(id) + prefix_;
        __builtin_prefetch(p, 0, 3);                       // eth/ip/udp header line
        __builtin_prefetch(p + 64, 0, 3);                  // mold header + first msg
    }
    void release(uint32_t id) noexcept { free_[nfree_++] = id; }

    // Refill: ef_vi_receive_init per free descriptor, then ONE doorbell
    // (ef_vi_receive_push). refill_batch is the amortization threshold that
    // gates the call, not a cap on how many are posted.
    void refill(bool) noexcept {
        uint32_t posted = 0;
        while (nfree_ > 0 && ef_vi_receive_space(&vi_) > 0) {
            const uint32_t id = free_[--nfree_];
            if (ef_vi_receive_init(&vi_, ef_memreg_dma_addr(&mr_, static_cast<size_t>(id) * buf_size_), id) < 0) {
                free_[nfree_++] = id; break;
            }
            ++posted;
        }
        if (posted) ef_vi_receive_push(&vi_);
    }

    template <class Deliver>
    void handle(uint32_t id, uint32_t bytes, Deliver& deliver) noexcept {
        const uint8_t* raw = buf(id);
        const uint8_t* f = raw + prefix_;
        ++st_.pkts; st_.bytes += bytes;
        if (bytes < wire::kUdpPayloadOff + wire::kMoldHdr) [[unlikely]] { ++st_.short_pkt; return; }
        if (!cls_.ok(f)) [[unlikely]] { ++st_.bad_hdr; return; }
        const uint64_t key = (uint64_t(wire::be32(f + wire::kIpDstOff)) << 16)
                           | wire::be16(f + wire::kUdpDstOff);
        const int chan = table_.lookup(key);
        if (chan < 0) [[unlikely]] { ++st_.unknown_dst; return; }
        const uint32_t udp_len = wire::be16(f + wire::kUdpLenOff);
        if (udp_len < 8 || wire::kUdpPayloadOff - 8 + udp_len > bytes) [[unlikely]] { ++st_.short_pkt; return; }
        uint64_t ts_ns = 0;
        if (cfg_.hw_timestamps) {
            struct timespec ts; unsigned fl;
            if (ef_vi_receive_get_timestamp_with_sync_flags(&vi_, raw, &ts, &fl) == 0)
                ts_ns = static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
        }
        deliver(static_cast<uint8_t>(chan), side_, f + wire::kUdpPayloadOff, udp_len - 8, ts_ns);
    }

    uint8_t* alloc_dma_impl(size_t bytes) noexcept {
        const size_t huge = 2u << 20;
        const size_t sz = (bytes + huge - 1) & ~(huge - 1);
        void* p = mmap(nullptr, sz, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p != MAP_FAILED) { huge_ = true; mem_size_ = sz; return static_cast<uint8_t*>(p); }
        huge_ = false;
        void* q = nullptr;
        if (posix_memalign(&q, 4096, bytes) != 0) return nullptr;
        return static_cast<uint8_t*>(q);
    }

    Config cfg_{};
    ef_driver_handle dh_{};
    ef_pd pd_{};
    ef_vi vi_{};
    ef_memreg mr_{};
    uint8_t* mem_ = nullptr; size_t mem_size_ = 0; bool huge_ = false;
    uint32_t nbufs_ = 0, buf_size_ = 0, prefix_ = 0;
    uint32_t* free_ = nullptr; uint32_t nfree_ = 0;
    ChannelTable table_;
    HeaderClassifier cls_;
    Stats st_;
    uint8_t side_ = 0;
    bool open_ = false;
};

// ---------------------------------------------------------------- receiver
template <class Sink, unsigned kMaxPorts = 4, unsigned kMaxChannels = 64>
class Receiver {
public:
    explicit Receiver(Sink& sink) : sink_(sink) {}

    // returns port index or <0
    int add_port(int ifindex, uint8_t side, const Config& cfg) {
        if (nports_ == kMaxPorts) return -1;
        const int rc = ports_[nports_].open(ifindex, side, cfg);
        return rc < 0 ? rc : static_cast<int>(nports_++);
    }
    int add_channel(uint8_t chan, unsigned port_idx, uint32_t group_ip_host, uint16_t port_host) {
        if (chan >= kMaxChannels || port_idx >= nports_) return -1;
        return ports_[port_idx].add_channel(chan, group_ip_host, port_host);
    }
    Port& port(unsigned i) noexcept { return ports_[i]; }
    const Stats& stats() const noexcept { return st_; }
    const Arbiter& arbiter(uint8_t chan) const noexcept { return arb_[chan]; }

    // one round over all ports; returns events handled
    int poll_once() noexcept {
        auto deliver = [this](uint8_t chan, uint8_t side, const uint8_t* payload,
                              uint32_t len, uint64_t ts) noexcept {
            arb_[chan].on_packet(chan, side, payload, len, ts, sink_, st_);
        };
        int n = 0;
        for (unsigned i = 0; i < nports_; ++i) n += ports_[i].poll(deliver);
        return n;
    }

private:
    Sink& sink_;
    Port ports_[kMaxPorts];
    unsigned nports_ = 0;
    Arbiter arb_[kMaxChannels];
    Stats st_;
};

}  // namespace md::net
