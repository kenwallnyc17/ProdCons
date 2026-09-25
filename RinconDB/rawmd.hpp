#pragma once
// rawmd.hpp — hybrid raw L1 feed handler + in-memory raw tick store.
//
// C++23, GCC 14.2, x86-64 with AVX-512 F/BW/VL + BMI2 + SSE4.2
//   g++ -std=c++23 -O3 -march=sapphirerapids   (or -march=znver4 / znver5)
//
// Threading model
//   * One LineHandler (LH) per pinned core. Each LH owns K queues; it is the only writer.
//   * Queues are append-only logs: records are never popped or overwritten, so any number of
//     client-handler threads may read concurrently (SPMC). Readers never slow the writer.
//   * Global queue index gq = lh_id * K + k.
//   * Hot path makes no syscalls, no locks, no heap allocation (heap is the pool's last resort).

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

#include <immintrin.h>
#include <sys/mman.h>

namespace rawmd {

inline constexpr std::size_t kCacheLine = 64;
inline constexpr std::size_t kHugePage  = std::size_t{2} << 20;

constexpr std::size_t round_up(std::size_t v, std::size_t a) noexcept { return (v + a - 1) / a * a; }

[[gnu::always_inline]] inline void cpu_relax() noexcept { _mm_pause(); }

// Adaptive wait for client-handler threads (never used on LH cores).
struct Backoff {
  uint32_t n = 0;
  void reset() noexcept { n = 0; }
  void wait() noexcept {
    if (n < 512) cpu_relax();
    else if (n < 2048) std::this_thread::yield();
    else std::this_thread::sleep_for(std::chrono::microseconds(50));
    ++n;
  }
};

// ================================ Symbol key =================================
// Up to 16 symbol bytes (CTA/UTP symbols fit in 11). Wire padding (space or NUL) is
// normalized to NUL so the feed side and the client request produce identical keys.
struct alignas(16) SymKey {
  uint64_t lo{0}, hi{0};
  bool operator==(const SymKey&) const = default;
  __m128i vec() const noexcept { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(this)); }
};

// Masked load: bytes outside [p, p+n) are never accessed (fault-suppressed), so this is
// safe at the very end of an RX buffer.
[[gnu::always_inline]] inline SymKey make_key(const void* p, unsigned n) noexcept {
  const auto lm = static_cast<__mmask16>(_bzhi_u32(0xFFFFu, n));
  __m128i v = _mm_maskz_loadu_epi8(lm, p);
  const __mmask16 sp = _mm_cmpeq_epi8_mask(v, _mm_set1_epi8(' '));
  v = _mm_mask_mov_epi8(v, sp, _mm_setzero_si128());
  SymKey k;
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&k), v);
  return k;
}

[[gnu::always_inline]] inline uint32_t hash_key(const SymKey& k) noexcept {
  return static_cast<uint32_t>(_mm_crc32_u64(_mm_crc32_u64(0x9E3779B9u, k.lo), k.hi));
}

// ================================= SymQMap ===================================
// symbol -> {global queue, symid}. Fixed capacity (sized at startup, no rehash), linear
// probing over 4-slot groups. The 4 keys of a group are one cache line and are compared
// against the target with a single AVX-512 compare.
//
// symid is the slot's position in the table: dense, unique, stable, and it is what the
// queue's per-record symbol sidecar stores (4 bytes instead of 16).
//
// Concurrency: many readers (every LH on every message, client handlers); inserts only by
// the LH that owns the symbol (a symbol lives on exactly one channel/LH). Slot claim is a
// CAS on the value word, so concurrent inserts of *different* symbols by different LHs are
// safe. Key is written with one aligned 16-byte store (single-copy atomic on AVX-capable
// x86), then the value is released. The racy SIMD key read is a formal C++ data race but
// well-defined on x86; a match is only trusted after an acquire load of the value.
class SymQMap {
 public:
  struct Entry { uint32_t gq; uint32_t symid; };

  explicit SymQMap(std::size_t max_symbols) {
    const std::size_t groups = std::bit_ceil(std::max<std::size_t>(max_symbols / 2, 16));  // <=50% load
    groups_ = std::make_unique<Group[]>(groups);
    gmask_  = groups - 1;
  }

  [[gnu::always_inline]] std::optional<Entry> find(const SymKey& k) const noexcept {
    const __m512i tgt = _mm512_broadcast_i32x4(k.vec());
    std::size_t g = hash_key(k) & gmask_;
    for (std::size_t probe = 0; probe <= gmask_; ++probe, g = (g + 1) & gmask_) {
      const Group& G = groups_[g];
      const unsigned eq = _mm512_cmpeq_epi64_mask(_mm512_load_si512(G.keys), tgt);
      if (const unsigned hit = eq & (eq >> 1) & 0x55u) [[likely]] {  // both 8-byte halves equal
        const unsigned s = static_cast<unsigned>(std::countr_zero(hit)) >> 1;
        uint64_t v;
        while ((v = G.vals[s].load(std::memory_order_acquire)) == kBusy) cpu_relax();
        return Entry{static_cast<uint32_t>(v - kBase), static_cast<uint32_t>(g * 4 + s)};
      }
      for (const auto& a : G.vals)
        if (a.load(std::memory_order_relaxed) == kEmpty) return std::nullopt;  // end of chain
    }
    return std::nullopt;
  }

  // Owning LH only (or single-threaded pre-open preload).
  Entry insert(const SymKey& k, uint32_t gq) noexcept {
    std::size_t g = hash_key(k) & gmask_;
    for (std::size_t probe = 0; probe <= gmask_; ++probe, g = (g + 1) & gmask_) {
      Group& G = groups_[g];
      for (unsigned s = 0; s < 4; ++s) {
        uint64_t v = G.vals[s].load(std::memory_order_acquire);
        if (v == kEmpty &&
            G.vals[s].compare_exchange_strong(v, kBusy, std::memory_order_acq_rel)) {
          _mm_store_si128(reinterpret_cast<__m128i*>(&G.keys[s]), k.vec());
          G.vals[s].store(uint64_t{gq} + kBase, std::memory_order_release);
          return Entry{gq, static_cast<uint32_t>(g * 4 + s)};
        }
        if (v >= kBase && G.keys[s] == k) return Entry{static_cast<uint32_t>(v - kBase),
                                                       static_cast<uint32_t>(g * 4 + s)};
      }
    }
    std::fputs("rawmd: SymQMap full — size max_symbols larger\n", stderr);
    std::abort();
  }

 private:
  static constexpr uint64_t kEmpty = 0, kBusy = 1, kBase = 2;
  struct alignas(128) Group {           // line 0: 4 keys, line 1: 4 values
    SymKey keys[4];
    std::atomic<uint64_t> vals[4];
  };
  std::unique_ptr<Group[]> groups_;
  std::size_t gmask_{0};
};

// ================================ ChunkPool ==================================
// Preallocated, prefaulted, hugepage-backed chunk arena. acquire() is a lock-free bump
// (fetch_add); chunks live for the session. Past the arena: aligned heap allocation.
class ChunkPool {
 public:
  ChunkPool(std::size_t chunk_bytes, std::size_t n_chunks)
      : chunk_bytes_(round_up(chunk_bytes, kHugePage)), n_(n_chunks) {
    const std::size_t bytes = chunk_bytes_ * n_;
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
    if (p != MAP_FAILED) {
      raw_ = p; raw_bytes_ = bytes; base_ = static_cast<std::byte*>(p); hugetlb_ = true;
    } else {  // no hugetlbfs reservation: THP, 2 MB aligned, prefault now instead of on the hot path
      raw_bytes_ = bytes + kHugePage;
      p = ::mmap(nullptr, raw_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (p == MAP_FAILED) throw std::bad_alloc();
      raw_  = p;
      base_ = reinterpret_cast<std::byte*>(
          (reinterpret_cast<uintptr_t>(p) + kHugePage - 1) & ~(kHugePage - 1));
      ::madvise(base_, bytes, MADV_HUGEPAGE);
      std::memset(base_, 0, bytes);
    }
  }
  ~ChunkPool() {
    ::munmap(raw_, raw_bytes_);
    for (void* h : heap_) std::free(h);
  }
  ChunkPool(const ChunkPool&) = delete;
  ChunkPool& operator=(const ChunkPool&) = delete;

  std::byte* acquire() noexcept {
    const std::size_t i = next_.fetch_add(1, std::memory_order_relaxed);
    if (i < n_) [[likely]] return base_ + i * chunk_bytes_;
    return heap_acquire();
  }

  std::size_t chunk_bytes() const noexcept { return chunk_bytes_; }
  std::size_t arena_used() const noexcept { return std::min(next_.load(std::memory_order_relaxed), n_); }
  std::size_t arena_chunks() const noexcept { return n_; }
  std::size_t heap_fallbacks() const noexcept { return fallbacks_.load(std::memory_order_relaxed); }
  bool hugetlb() const noexcept { return hugetlb_; }

 private:
  [[gnu::cold, gnu::noinline]] std::byte* heap_acquire() noexcept {
    void* h = std::aligned_alloc(kHugePage, chunk_bytes_);
    if (!h) return nullptr;
    ::madvise(h, chunk_bytes_, MADV_HUGEPAGE);
    { std::lock_guard g(heap_mu_); heap_.push_back(h); }
    fallbacks_.fetch_add(1, std::memory_order_relaxed);
    return static_cast<std::byte*>(h);
  }

  std::size_t chunk_bytes_, n_;
  void* raw_{nullptr};
  std::size_t raw_bytes_{0};
  std::byte* base_{nullptr};
  bool hugetlb_{false};
  alignas(kCacheLine) std::atomic<std::size_t> next_{0};
  alignas(kCacheLine) std::atomic<std::size_t> fallbacks_{0};
  std::mutex heap_mu_;
  std::vector<void*> heap_;
};

// ============================== Chunk layout =================================
// One chunk = [ts[kCap] u64][symid[kCap] u32][records[kCap] x ElemSize].
// The ts / symid sidecars are dense SoA arrays so replay can binary-search time and
// SIMD-filter by symbol (16 records per compare) without touching record memory.
template <std::size_t ElemSize, unsigned ChunkShift>
struct Layout {
  static_assert(ElemSize % 64 == 0 && ElemSize >= 64, "ElemSize must be a multiple of 64");
  static constexpr std::size_t kElem       = ElemSize;
  static constexpr unsigned    kShift      = ChunkShift;
  static constexpr std::size_t kCap        = std::size_t{1} << ChunkShift;
  static constexpr std::size_t kMask       = kCap - 1;
  static constexpr std::size_t kHdr        = 8;
  static constexpr std::size_t kMaxPayload = ElemSize - kHdr;
  static constexpr std::size_t kTsOff      = 0;
  static constexpr std::size_t kSymOff     = kCap * sizeof(uint64_t);
  static constexpr std::size_t kRecOff     = round_up(kSymOff + kCap * sizeof(uint32_t), kCacheLine);
  static constexpr std::size_t kBytes      = round_up(kRecOff + kCap * ElemSize, kHugePage);
};
// 128 B records, 64K per chunk -> 10 MB chunks, ~8% slack.
using DefaultLayout = Layout<128, 16>;

struct RecHdr {          // first 8 bytes of every record
  uint16_t len;          // raw message length
  uint8_t  type;         // protocol message type
  uint8_t  flags;
  uint32_t aux;          // free (e.g. packet seq for gap forensics)
};
static_assert(sizeof(RecHdr) == 8);

// Build the record in registers and write whole cache lines with aligned stores.
// Masked loads never read past src+len (fault-suppressed), so no source over-read.
template <std::size_t E>
[[gnu::always_inline]] inline void store_record(std::byte* dst, uint64_t hdr, const uint8_t* src,
                                                uint32_t len) noexcept {
  const uint32_t n0 = std::min<uint32_t>(len, 56);
  const __mmask64 m0 = _bzhi_u64(~0ull, n0) << 8;  // lanes 8..8+n0 <- src[0..n0)
  __m512i v = _mm512_maskz_loadu_epi8(
      m0, reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(src) - 8));
  v = _mm512_mask_set1_epi64(v, 1, static_cast<long long>(hdr));
  _mm512_store_si512(dst, v);
  if constexpr (E > 64) {
    for (uint32_t off = 56, line = 1; off < len && line < E / 64; off += 64, ++line) {
      const __mmask64 m = _bzhi_u64(~0ull, std::min<uint32_t>(len - off, 64));
      _mm512_store_si512(dst + line * 64, _mm512_maskz_loadu_epi8(m, src + off));
    }
  }
}

// ================================ MsgQueue ===================================
// The per-LH "spsc queue": an append-only, chunk-grown log. Record i lives at
// chunk dir_[i >> kShift], slot i & kMask. The chunk directory is a fixed array, so
// growth never moves anything a reader may be looking at.
template <class L>
class MsgQueue {
 public:
  using layout = L;

  void init(ChunkPool& pool, std::size_t max_chunks) {
    if (pool.chunk_bytes() < L::kBytes) throw std::invalid_argument("pool chunk smaller than layout");
    pool_       = &pool;
    max_chunks_ = max_chunks;
    dir_        = std::make_unique<std::atomic<std::byte*>[]>(max_chunks);
    cur_        = pool.acquire();
    if (!cur_) throw std::bad_alloc();
    dir_[0].store(cur_, std::memory_order_relaxed);
  }

  // ---- producer: owning LH thread only ----
  // Writes the record; not visible to readers until publish().
  [[gnu::always_inline]] bool append(uint32_t symid, uint64_t ts, const uint8_t* msg, uint32_t len,
                                     uint8_t type) noexcept {
    const uint64_t i = w_;
    const std::size_t s = i & L::kMask;
    if (s == 0 && i != 0) [[unlikely]] {
      if (!advance(i >> L::kShift)) return false;
    }
    std::byte* c = cur_;
    reinterpret_cast<uint64_t*>(c + L::kTsOff)[s]  = ts;
    reinterpret_cast<uint32_t*>(c + L::kSymOff)[s] = symid;
    const RecHdr h{static_cast<uint16_t>(len), type, 0, 0};
    store_record<L::kElem>(c + L::kRecOff + s * L::kElem, std::bit_cast<uint64_t>(h), msg, len);
    if (s == L::kCap / 2) [[unlikely]] prepare((i >> L::kShift) + 1);  // next chunk ready early
    w_ = i + 1;
    return true;
  }
  void publish() noexcept { pub_.store(w_, std::memory_order_release); }
  uint64_t written() const noexcept { return w_; }
  uint64_t dropped() const noexcept { return dropped_; }

  // ---- consumers: any thread ----
  uint64_t committed() const noexcept { return pub_.load(std::memory_order_acquire); }
  // Valid for any chunk holding an index < committed() (ordered by the release in publish()).
  const std::byte* chunk(uint64_t c) const noexcept { return dir_[c].load(std::memory_order_relaxed); }
  static const uint64_t* ts_of(const std::byte* c) noexcept {
    return reinterpret_cast<const uint64_t*>(c + L::kTsOff);
  }
  static const uint32_t* sym_of(const std::byte* c) noexcept {
    return reinterpret_cast<const uint32_t*>(c + L::kSymOff);
  }
  static const std::byte* rec_of(const std::byte* c, std::size_t s) noexcept {
    return c + L::kRecOff + s * L::kElem;
  }
  uint64_t ts_at(uint64_t i) const noexcept { return ts_of(chunk(i >> L::kShift))[i & L::kMask]; }

 private:
  bool advance(uint64_t c) noexcept {
    if (c >= max_chunks_) [[unlikely]] { ++dropped_; return false; }
    std::byte* n = next_ ? next_ : pool_->acquire();
    if (!n) [[unlikely]] { ++dropped_; return false; }
    next_ = nullptr;
    cur_  = n;
    dir_[c].store(n, std::memory_order_relaxed);
    return true;
  }
  void prepare(uint64_t c) noexcept {
    if (c < max_chunks_ && !next_) next_ = pool_->acquire();
  }

  // read-mostly, shared with readers
  alignas(kCacheLine) std::unique_ptr<std::atomic<std::byte*>[]> dir_;
  std::size_t max_chunks_{0};
  // producer-private
  alignas(kCacheLine) uint64_t w_{0};
  std::byte* cur_{nullptr};
  std::byte* next_{nullptr};
  ChunkPool* pool_{nullptr};
  uint64_t dropped_{0};
  // published append index (the only line readers poll)
  alignas(kCacheLine) std::atomic<uint64_t> pub_{0};
  char pad_[kCacheLine - sizeof(std::atomic<uint64_t>)];
};

// ================================ QToTimeIdx =================================
// idx[gq][b] = index of the first record in queue gq with ts >= bucket_start(b).
// Written by the owning LH when its (monotone) feed time crosses a bucket boundary,
// so there is no timer thread and no cross-thread read of the writer's index.
class TimeIndex {
 public:
  TimeIndex(uint64_t start_ns, uint64_t bucket_ns, uint32_t n_buckets, uint32_t n_lh, uint32_t q_per_lh)
      : start_(start_ns), bucket_(bucket_ns), nb_(n_buckets),
        idx_(std::make_unique<std::atomic<uint64_t>[]>(std::size_t{n_lh} * q_per_lh * n_buckets)),
        pub_(std::make_unique<PubSlot[]>(n_lh)) {}

  uint64_t start() const noexcept { return start_; }
  uint32_t n_buckets() const noexcept { return nb_; }
  uint64_t bucket_start(uint32_t b) const noexcept { return start_ + uint64_t{b} * bucket_; }
  uint32_t bucket_of(uint64_t ts) const noexcept {
    if (ts < start_) return 0;
    return static_cast<uint32_t>(std::min<uint64_t>((ts - start_) / bucket_, nb_ - 1));
  }
  void set(uint32_t gq, uint32_t b, uint64_t i) noexcept {
    idx_[std::size_t{gq} * nb_ + b].store(i, std::memory_order_relaxed);
  }
  uint64_t get(uint32_t gq, uint32_t b) const noexcept {
    return idx_[std::size_t{gq} * nb_ + b].load(std::memory_order_relaxed);
  }
  void publish(uint32_t lh, int32_t b) noexcept { pub_[lh].v.store(b, std::memory_order_release); }
  int32_t published(uint32_t lh) const noexcept { return pub_[lh].v.load(std::memory_order_acquire); }

 private:
  struct alignas(kCacheLine) PubSlot { std::atomic<int32_t> v{-1}; };
  uint64_t start_, bucket_;
  uint32_t nb_;
  std::unique_ptr<std::atomic<uint64_t>[]> idx_;
  std::unique_ptr<PubSlot[]> pub_;
};

// ================================== Store ====================================
template <class Q>
struct Store {
  struct Config {
    uint32_t    n_lh;
    uint32_t    q_per_lh;          // K (<= 64)
    std::size_t pool_chunks;       // preallocated arena chunks (>= n_lh*K)
    std::size_t max_chunks_per_q;  // directory size per queue
    std::size_t max_symbols;
    uint64_t    session_start_ns;
    uint64_t    bucket_ns;         // N seconds
    uint32_t    n_buckets;
  };

  explicit Store(const Config& c)
      : cfg(c),
        pool(Q::layout::kBytes, c.pool_chunks),
        symmap(c.max_symbols),
        time(c.session_start_ns, c.bucket_ns, c.n_buckets, c.n_lh, c.q_per_lh),
        queues(std::make_unique<Q[]>(std::size_t{c.n_lh} * c.q_per_lh)) {
    for (std::size_t i = 0; i < std::size_t{c.n_lh} * c.q_per_lh; ++i)
      queues[i].init(pool, c.max_chunks_per_q);
  }
  uint32_t lh_of(uint32_t gq) const noexcept { return gq / cfg.q_per_lh; }

  Config cfg;
  ChunkPool pool;
  SymQMap symmap;
  TimeIndex time;
  std::unique_ptr<Q[]> queues;
};

// =============================== LineHandler =================================
// Proto contract (see toy_l1.hpp):
//   template<class F> static void for_each_msg(const uint8_t* pkt, size_t len, F&& f);
//       f(const uint8_t* msg, uint32_t msg_len, uint8_t type, const SymKey& key)
//   static bool decode(const uint8_t* msg, uint32_t len, uint8_t type, uint64_t rx_ns, Out&);
template <class Proto, class Q>
class LineHandler {
  using L = typename Q::layout;

 public:
  struct Stats { uint64_t pkts{0}, msgs{0}, new_symbols{0}, oversize{0}, dropped{0}; };

  LineHandler(uint32_t lh_id, Store<Q>& st)
      : st_(st), id_(lh_id), k_(st.cfg.q_per_lh), base_(lh_id * st.cfg.q_per_lh),
        q_(&st.queues[base_]), next_boundary_(st.time.start()) {
    if (k_ == 0 || k_ > 64) throw std::invalid_argument("q_per_lh must be 1..64");
  }

  // Pre-open: pin a symbol to local queue k (e.g. bin-packed by prior-day volume).
  SymQMap::Entry pin(const SymKey& key, uint32_t k) { return st_.symmap.insert(key, base_ + k % k_); }

  // One arbitrated packet. rx_ns: NIC hardware or TSC receive time, same domain as client requests.
  void on_packet(const uint8_t* pkt, std::size_t len, uint64_t rx_ns) noexcept {
    rx_ns    = std::max(rx_ns, last_rx_);  // binary search relies on monotone per-LH time
    last_rx_ = rx_ns;
    if (rx_ns >= next_boundary_) [[unlikely]] roll(rx_ns);
    ++stats_.pkts;

    uint64_t dirty = 0;
    Proto::for_each_msg(pkt, len, [&](const uint8_t* m, uint32_t mlen, uint8_t type, const SymKey& key) {
      if (mlen > L::kMaxPayload) [[unlikely]] { ++stats_.oversize; return; }
      const SymQMap::Entry e = resolve(key);
      const uint32_t k = e.gq - base_;
      if (!q_[k].append(e.symid, rx_ns, m, mlen, type)) [[unlikely]] ++stats_.dropped;
      dirty |= uint64_t{1} << k;
      ++stats_.msgs;
    });
    for (; dirty; dirty &= dirty - 1) q_[std::countr_zero(dirty)].publish();  // one release per queue per packet
  }

  const Stats& stats() const noexcept { return stats_; }

 private:
  [[gnu::always_inline]] SymQMap::Entry resolve(const SymKey& key) noexcept {
    if (const auto e = st_.symmap.find(key)) [[likely]] return *e;
    return assign(key);
  }
  [[gnu::cold, gnu::noinline]] SymQMap::Entry assign(const SymKey& key) noexcept {
    ++stats_.new_symbols;
    return st_.symmap.insert(key, base_ + (rr_++ % k_));
  }
  [[gnu::cold, gnu::noinline]] void roll(uint64_t rx_ns) noexcept {
    TimeIndex& ti = st_.time;
    const uint32_t b = ti.bucket_of(rx_ns);
    for (int64_t bb = cur_bucket_ + 1; bb <= int64_t{b}; ++bb)   // also fills gaps (quiet periods)
      for (uint32_t k = 0; k < k_; ++k) ti.set(base_ + k, static_cast<uint32_t>(bb), q_[k].written());
    cur_bucket_ = b;
    ti.publish(id_, static_cast<int32_t>(b));
    next_boundary_ = (b + 1 < ti.n_buckets()) ? ti.bucket_start(b + 1) : UINT64_MAX;
  }

  Store<Q>& st_;
  const uint32_t id_, k_, base_;
  Q* const q_;
  uint64_t next_boundary_;
  uint64_t last_rx_{0};
  int64_t cur_bucket_{-1};
  uint32_t rr_{0};
  Stats stats_;
};

// ================================ Replayer ===================================
// Read-side logic for one symbol. No I/O; any thread.
template <class Q>
class Replayer {
  using L = typename Q::layout;

 public:
  Replayer(const Store<Q>& st, SymQMap::Entry e)
      : st_(st), q_(st.queues[e.gq]), gq_(e.gq), symid_(e.symid) {}

  const Q& queue() const noexcept { return q_; }

  // First record index in the queue with ts >= t (all symbols; filter happens in scan).
  // QToTimeIdx narrows to one bucket; the ts sidecar makes the answer exact.
  uint64_t find_start(uint64_t t) const noexcept {
    const TimeIndex& ti = st_.time;
    const int32_t pb = ti.published(st_.lh_of(gq_));  // read before committed(): idx[<=pb] <= w
    const uint64_t w = q_.committed();
    uint64_t lo = 0, hi = w;
    if (pb >= 0 && t >= ti.start()) {
      const uint32_t b = ti.bucket_of(t);
      if (static_cast<int32_t>(b) <= pb) {
        lo = ti.get(gq_, b);
        if (static_cast<int32_t>(b) < pb) hi = ti.get(gq_, b + 1);
      } else {
        lo = ti.get(gq_, static_cast<uint32_t>(pb));
      }
    }
    hi = std::min(hi, w);
    lo = std::min(lo, hi);
    while (lo < hi) {
      const uint64_t mid = lo + (hi - lo) / 2;
      if (q_.ts_at(mid) < t) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  // Last record of this symbol strictly before index `end` (the prevailing L1 state at t),
  // scanning at most max_scan queue records backwards. 16 records per AVX-512 compare.
  std::optional<uint64_t> find_prior(uint64_t end, uint64_t max_scan) const noexcept {
    const __m512i tgt = _mm512_set1_epi32(static_cast<int>(symid_));
    const uint64_t stop = end > max_scan ? end - max_scan : 0;
    uint64_t i = end;
    while (i > stop) {
      const uint64_t cbase = ((i - 1) >> L::kShift) << L::kShift;
      const uint64_t lo = std::max(cbase, stop);
      const uint32_t* sym = Q::sym_of(q_.chunk(cbase >> L::kShift));
      while (i > lo) {
        const auto n = static_cast<unsigned>(std::min<uint64_t>(16, i - lo));
        const auto s = static_cast<uint32_t>(i - n - cbase);
        const auto lm = static_cast<__mmask16>(_bzhi_u32(0xFFFFu, n));
        const unsigned m = _mm512_mask_cmpeq_epi32_mask(lm, _mm512_maskz_loadu_epi32(lm, sym + s), tgt);
        if (m) return cbase + s + (31u - static_cast<unsigned>(std::countl_zero(m)));
        i -= n;
      }
    }
    return std::nullopt;
  }

  // f(const RecHdr&, const uint8_t* payload, uint64_t ts) for each record of this symbol in [from, to).
  template <class F>
  void scan(uint64_t from, uint64_t to, F&& f) const {
    const __m512i tgt = _mm512_set1_epi32(static_cast<int>(symid_));
    while (from < to) {
      const uint64_t c = from >> L::kShift;
      const uint64_t cbase = c << L::kShift;
      const uint64_t cend = std::min<uint64_t>(to, cbase + L::kCap);
      const std::byte* ch = q_.chunk(c);
      const uint32_t* sym = Q::sym_of(ch);
      const uint64_t* ts = Q::ts_of(ch);
      for (uint64_t i = from; i < cend; i += 16) {
        const auto s = static_cast<uint32_t>(i - cbase);
        const auto n = static_cast<unsigned>(std::min<uint64_t>(16, cend - i));
        const auto lm = static_cast<__mmask16>(_bzhi_u32(0xFFFFu, n));
        _mm_prefetch(reinterpret_cast<const char*>(sym + s + 256), _MM_HINT_T0);
        unsigned m = _mm512_mask_cmpeq_epi32_mask(lm, _mm512_maskz_loadu_epi32(lm, sym + s), tgt);
        for (; m; m &= m - 1) {
          const uint32_t j = s + static_cast<uint32_t>(std::countr_zero(m));
          const std::byte* r = Q::rec_of(ch, j);
          f(*reinterpret_cast<const RecHdr*>(r), reinterpret_cast<const uint8_t*>(r + L::kHdr), ts[j]);
        }
      }
      from = cend;
    }
  }

  template <class F>
  void visit(uint64_t i, F&& f) const {
    const std::byte* ch = q_.chunk(i >> L::kShift);
    const std::size_t s = i & L::kMask;
    const std::byte* r = Q::rec_of(ch, s);
    f(*reinterpret_cast<const RecHdr*>(r), reinterpret_cast<const uint8_t*>(r + L::kHdr), Q::ts_of(ch)[s]);
  }

 private:
  const Store<Q>& st_;
  const Q& q_;
  uint32_t gq_, symid_;
};

}  // namespace rawmd
