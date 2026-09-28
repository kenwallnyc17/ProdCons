#pragma once
// rawmd.hpp — hybrid raw L1 feed handler + tick store (v2).
//
// C++23, GCC 14.2, x86-64 AVX-512 F/BW/VL + BMI2 + SSE4.2  (-march=sapphirerapids | znver4 | znver5)
//
// One binary per feed: Layout<ChunkShift, MaxMsg, RecAlign> fixes the record stride at
// compile time (16-byte header + MaxMsg, rounded to RecAlign). No per-record offsets.
//
// Design notes
//   * Search column is the feed's own timestamp (extracted alongside the symbol on ingest);
//     receive time is kept in the record header.
//   * Backing is RAM (anonymous, THP) or a file on NVMe (MAP_SHARED): the kernel page cache
//     is the RAM tier. A provisioner thread pre-faults + mlocks the LH's next chunk so the
//     hot path never takes a page fault, and unlocks sealed chunks so they can be evicted.
//   * Several feeds in one store: every record carries a proto id; decoders are registered
//     per proto so one client can subscribe across feeds.
//   * Per-LH watermark (latest published feed time) lets a client merge several queues in
//     time order, in history and in the live tail.
//
// Threading: one LineHandler per pinned core, sole writer of its K queues. Queues are
// append-only logs read by any number of client threads. Hot path: no syscalls, no locks,
// no page faults, no heap (heap is the pool's last resort).

#include <algorithm>
#include <array>
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
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23  // Linux 5.14+
#endif

namespace rawmd {

inline constexpr std::size_t kCacheLine = 64;
inline constexpr std::size_t kHugePage  = std::size_t{2} << 20;

constexpr std::size_t round_up(std::size_t v, std::size_t a) noexcept { return (v + a - 1) / a * a; }
[[gnu::always_inline]] inline void cpu_relax() noexcept { _mm_pause(); }

struct Backoff {  // client threads only
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
struct alignas(16) SymKey {
  uint64_t lo{0}, hi{0};
  bool operator==(const SymKey&) const = default;
  __m128i vec() const noexcept { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(this)); }
};

// Up to 16 bytes; trailing spaces/NULs stripped (interior spaces kept). Masked load never
// touches bytes outside [p, p+n).
[[gnu::always_inline]] inline SymKey make_key(const void* p, unsigned n) noexcept {
  const auto lm = static_cast<__mmask16>(_bzhi_u32(0xFFFFu, n));
  const __m128i v = _mm_maskz_loadu_epi8(lm, p);
  const unsigned nonpad = static_cast<unsigned>(lm) &
                          ~static_cast<unsigned>(_mm_cmpeq_epi8_mask(v, _mm_set1_epi8(' '))) &
                          ~static_cast<unsigned>(_mm_cmpeq_epi8_mask(v, _mm_setzero_si128()));
  const unsigned keep = nonpad ? _bzhi_u32(0xFFFFu, 32u - static_cast<unsigned>(std::countl_zero(nonpad))) : 0u;
  SymKey k;
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&k), _mm_maskz_mov_epi8(static_cast<__mmask16>(keep), v));
  return k;
}

[[gnu::always_inline]] inline uint32_t hash_key(const SymKey& k) noexcept {
  return static_cast<uint32_t>(_mm_crc32_u64(_mm_crc32_u64(0x9E3779B9u, k.lo), k.hi));
}

// ================================= SymQMap ===================================
// symbol -> {global queue, symid}. Fixed capacity, linear probing over 4-slot groups; the
// four keys of a group are one cache line, matched with one AVX-512 compare. symid is the
// slot index (dense, stable). Inserts: CAS-claim value, 16 B key store, release value.
// Readers: racy SIMD key read (x86-defined), trusted only after an acquire of the value.
class SymQMap {
 public:
  struct Entry { uint32_t gq; uint32_t symid; };

  explicit SymQMap(std::size_t max_symbols) {
    const std::size_t groups = std::bit_ceil(std::max<std::size_t>(max_symbols / 2, 16));
    groups_ = std::make_unique<Group[]>(groups);
    gmask_  = groups - 1;
  }
  std::size_t slots() const noexcept { return (gmask_ + 1) * 4; }

  [[gnu::always_inline]] std::optional<Entry> find(const SymKey& k) const noexcept {
    const __m512i tgt = _mm512_broadcast_i32x4(k.vec());
    std::size_t g = hash_key(k) & gmask_;
    for (std::size_t probe = 0; probe <= gmask_; ++probe, g = (g + 1) & gmask_) {
      const Group& G = groups_[g];
      const unsigned eq = _mm512_cmpeq_epi64_mask(_mm512_load_si512(G.keys), tgt);
      if (const unsigned hit = eq & (eq >> 1) & 0x55u) [[likely]] {
        const unsigned s = static_cast<unsigned>(std::countr_zero(hit)) >> 1;
        uint64_t v;
        while ((v = G.vals[s].load(std::memory_order_acquire)) == kBusy) cpu_relax();
        return Entry{static_cast<uint32_t>(v - kBase), static_cast<uint32_t>(g * 4 + s)};
      }
      for (const auto& a : G.vals)
        if (a.load(std::memory_order_relaxed) == kEmpty) return std::nullopt;
    }
    return std::nullopt;
  }

  Entry insert(const SymKey& k, uint32_t gq) noexcept {  // owning LH (or pre-open preload)
    std::size_t g = hash_key(k) & gmask_;
    for (std::size_t probe = 0; probe <= gmask_; ++probe, g = (g + 1) & gmask_) {
      Group& G = groups_[g];
      for (unsigned s = 0; s < 4; ++s) {
        uint64_t v = G.vals[s].load(std::memory_order_acquire);
        if (v == kEmpty && G.vals[s].compare_exchange_strong(v, kBusy, std::memory_order_acq_rel)) {
          _mm_store_si128(reinterpret_cast<__m128i*>(&G.keys[s]), k.vec());
          G.vals[s].store(uint64_t{gq} + kBase, std::memory_order_release);
          return Entry{gq, static_cast<uint32_t>(g * 4 + s)};
        }
        if (v >= kBase && G.keys[s] == k)
          return Entry{static_cast<uint32_t>(v - kBase), static_cast<uint32_t>(g * 4 + s)};
      }
    }
    std::fputs("rawmd: SymQMap full — raise max_symbols\n", stderr);
    std::abort();
  }

 private:
  static constexpr uint64_t kEmpty = 0, kBusy = 1, kBase = 2;
  struct alignas(128) Group { SymKey keys[4]; std::atomic<uint64_t> vals[4]; };
  std::unique_ptr<Group[]> groups_;
  std::size_t gmask_{0};
};

// ================================ ChunkPool ==================================
// Arena of n chunks. Backing::Ram: anonymous, THP, fully prefaulted at startup.
// Backing::File: MAP_SHARED file (NVMe); pages are faulted in by the provisioner just
// before use and written back / evicted by the kernel. acquire() is a lock-free bump.
class ChunkPool {
 public:
  enum class Backing { Ram, File };

  ChunkPool(std::size_t chunk_bytes, std::size_t n_chunks, Backing backing, const std::string& path = {})
      : chunk_bytes_(round_up(chunk_bytes, kHugePage)), n_(n_chunks), backing_(backing) {
    const std::size_t bytes = chunk_bytes_ * n_;
    if (backing == Backing::File) {
      fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
      if (fd_ < 0 || ::ftruncate(fd_, static_cast<off_t>(bytes)) != 0) throw std::runtime_error("pool file");
      void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_, 0);
      if (p == MAP_FAILED) throw std::bad_alloc();
      raw_ = p; raw_bytes_ = bytes; base_ = static_cast<std::byte*>(p);
      return;
    }
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
    if (p != MAP_FAILED) { raw_ = p; raw_bytes_ = bytes; base_ = static_cast<std::byte*>(p); hugetlb_ = true; return; }
    raw_bytes_ = bytes + kHugePage;
    p = ::mmap(nullptr, raw_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::bad_alloc();
    raw_  = p;
    base_ = reinterpret_cast<std::byte*>((reinterpret_cast<uintptr_t>(p) + kHugePage - 1) & ~(kHugePage - 1));
    ::madvise(base_, bytes, MADV_HUGEPAGE);
    std::memset(base_, 0, bytes);
  }
  ~ChunkPool() {
    ::munmap(raw_, raw_bytes_);
    if (fd_ >= 0) ::close(fd_);
    for (void* h : heap_) std::free(h);
  }
  ChunkPool(const ChunkPool&) = delete;
  ChunkPool& operator=(const ChunkPool&) = delete;

  std::byte* acquire() noexcept {
    const std::size_t i = next_.fetch_add(1, std::memory_order_relaxed);
    if (i < n_) [[likely]] return base_ + i * chunk_bytes_;
    return heap_acquire();
  }
  // Provisioner thread: make the chunk fault-free for the LH.
  void prepare(std::byte* c) noexcept {
    ::madvise(c, chunk_bytes_, MADV_POPULATE_WRITE);
    if (::mlock(c, chunk_bytes_) != 0) mlock_fail_.fetch_add(1, std::memory_order_relaxed);
  }
  void release(std::byte* c) noexcept { ::munlock(c, chunk_bytes_); }

  std::size_t chunk_bytes() const noexcept { return chunk_bytes_; }
  std::size_t arena_used() const noexcept { return std::min(next_.load(std::memory_order_relaxed), n_); }
  std::size_t arena_chunks() const noexcept { return n_; }
  std::size_t heap_fallbacks() const noexcept { return fallbacks_.load(std::memory_order_relaxed); }
  std::size_t mlock_failures() const noexcept { return mlock_fail_.load(std::memory_order_relaxed); }
  Backing backing() const noexcept { return backing_; }
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
  Backing backing_;
  int fd_{-1};
  void* raw_{nullptr};
  std::size_t raw_bytes_{0};
  std::byte* base_{nullptr};
  bool hugetlb_{false};
  alignas(kCacheLine) std::atomic<std::size_t> next_{0};
  alignas(kCacheLine) std::atomic<std::size_t> fallbacks_{0}, mlock_fail_{0};
  std::mutex heap_mu_;
  std::vector<void*> heap_;
};

// ============================== Chunk layout =================================
struct RecHdr {          // 16 bytes, followed by the raw message
  uint16_t len;          // raw message length
  uint8_t  type;         // protocol message type
  uint8_t  proto;        // decoder id (see Store::decoders)
  uint32_t aux;          // free (packet seq, flags)
  uint64_t rx_ns;        // receive time
};
static_assert(sizeof(RecHdr) == 16);
inline constexpr std::size_t kRecHdr = 16;

// [ts[kCap] u64][sym[kCap] u32][records[kCap] x kRec]
// RecAlign = 8 packs records; RecAlign = 64 makes every record cache-line aligned at the
// cost of padding (worth it only if kRecHdr + MaxMsg is already close to a multiple of 64).
template <unsigned ChunkShift, std::size_t MaxMsg, std::size_t RecAlign = 8>
struct Layout {
  static_assert(std::has_single_bit(RecAlign) && RecAlign >= 8);
  static_assert(MaxMsg <= 0xFFFF);
  static constexpr unsigned    kShift   = ChunkShift;
  static constexpr std::size_t kCap     = std::size_t{1} << ChunkShift;
  static constexpr std::size_t kMask    = kCap - 1;
  static constexpr std::size_t kMaxMsg  = MaxMsg;
  static constexpr std::size_t kRec     = round_up(kRecHdr + MaxMsg, RecAlign);
  static constexpr std::size_t kTsOff   = 0;
  static constexpr std::size_t kSymOff  = kCap * 8;
  static constexpr std::size_t kDataOff = round_up(kSymOff + kCap * 4, kCacheLine);
  static constexpr std::size_t kBytes   = round_up(kDataOff + kCap * kRec, kHugePage);
};

// Header + payload assembled in zmm registers; masked loads never read past src+len.
[[gnu::always_inline]] inline void store_record(std::byte* dst, const RecHdr& h, const uint8_t* src, uint32_t len) noexcept {
  const uint32_t n0 = std::min<uint32_t>(len, 48);
  __m512i v = _mm512_maskz_loadu_epi8(_bzhi_u64(~0ull, n0) << 16,
                                      reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(src) - 16));
  uint64_t h0; std::memcpy(&h0, &h, 8);
  v = _mm512_mask_set1_epi64(v, 1, static_cast<long long>(h0));
  v = _mm512_mask_set1_epi64(v, 2, static_cast<long long>(h.rx_ns));
  _mm512_storeu_si512(dst, v);
  for (uint32_t off = 48; off < len; off += 64) {
    const __mmask64 m = _bzhi_u64(~0ull, std::min<uint32_t>(len - off, 64));
    _mm512_storeu_si512(dst + kRecHdr + off, _mm512_maskz_loadu_epi8(m, src + off));
  }
}

// ================================ MsgQueue ===================================
// Append-only, chunk-grown log. Record i: chunk i >> kShift, slot i & kMask.
// Directory is a fixed array, so growth never moves anything.
template <class L>
class MsgQueue {
 public:
  using layout = L;
  struct Stats { uint64_t dropped{0}, oversize{0}, late_chunks{0}; };

  void init(ChunkPool& pool, std::size_t max_chunks) {
    if (pool.chunk_bytes() < L::kBytes) throw std::invalid_argument("chunk smaller than layout");
    pool_ = &pool; max_chunks_ = max_chunks;
    dir_ = std::make_unique<std::atomic<std::byte*>[]>(max_chunks);
    cur_ = pool.acquire();
    if (!cur_) throw std::bad_alloc();
    pool.prepare(cur_);
    dir_[0].store(cur_, std::memory_order_relaxed);
  }

  // ---- producer: owning LH only ----
  [[gnu::always_inline]] bool append(uint32_t symid, uint64_t ts, const RecHdr& h, const uint8_t* msg) noexcept {
    if (h.len > L::kMaxMsg) [[unlikely]] { ++st_.oversize; return false; }
    const uint64_t i = w_;
    const std::size_t s = i & L::kMask;
    if (s == 0 && i != 0) [[unlikely]] {
      if (!advance(i >> L::kShift)) { ++st_.dropped; return false; }
    }
    std::byte* c = cur_;
    reinterpret_cast<uint64_t*>(c + L::kTsOff)[s]  = ts;
    reinterpret_cast<uint32_t*>(c + L::kSymOff)[s] = symid;
    store_record(c + L::kDataOff + s * L::kRec, h, msg, h.len);
    if (s == L::kCap / 2) [[unlikely]] request_next();
    w_ = i + 1;
    return true;
  }
  void publish() noexcept { pub_.store(w_, std::memory_order_release); }
  uint64_t written() const noexcept { return w_; }
  const Stats& stats() const noexcept { return st_; }

  // ---- provisioner thread ----
  void provision_step(std::size_t keep_locked) noexcept {
    if (!want_.load(std::memory_order_acquire)) return;
    std::byte* c = pool_->acquire();
    if (c) pool_->prepare(c);
    ready_.store(c, std::memory_order_release);
    want_.store(false, std::memory_order_release);
    locked_.push_back(c);
    while (locked_.size() > keep_locked) { if (locked_.front()) pool_->release(locked_.front()); locked_.erase(locked_.begin()); }
  }

  // ---- consumers: any thread; valid for indices < committed() ----
  uint64_t committed() const noexcept { return pub_.load(std::memory_order_acquire); }
  const std::byte* chunk(uint64_t c) const noexcept { return dir_[c].load(std::memory_order_relaxed); }
  static const uint64_t* ts_of(const std::byte* c) noexcept { return reinterpret_cast<const uint64_t*>(c + L::kTsOff); }
  static const uint32_t* sym_of(const std::byte* c) noexcept { return reinterpret_cast<const uint32_t*>(c + L::kSymOff); }
  static const std::byte* rec_of(const std::byte* c, std::size_t s) noexcept { return c + L::kDataOff + s * L::kRec; }
  uint64_t ts_at(uint64_t i) const noexcept { return ts_of(chunk(i >> L::kShift))[i & L::kMask]; }

 private:
  void request_next() noexcept {
    if (!next_ && !want_.load(std::memory_order_relaxed)) {
      next_ = ready_.exchange(nullptr, std::memory_order_acquire);
      if (!next_) want_.store(true, std::memory_order_release);
    }
  }
  bool advance(uint64_t c) noexcept {
    if (c >= max_chunks_) [[unlikely]] return false;
    std::byte* n = next_ ? next_ : ready_.exchange(nullptr, std::memory_order_acquire);
    if (!n) { n = pool_->acquire(); ++st_.late_chunks; }  // provisioner didn't make it: sync path
    if (!n) [[unlikely]] return false;
    next_ = nullptr; cur_ = n;
    dir_[c].store(n, std::memory_order_relaxed);
    return true;
  }

  alignas(kCacheLine) std::unique_ptr<std::atomic<std::byte*>[]> dir_;  // read-mostly
  std::size_t max_chunks_{0};
  ChunkPool* pool_{nullptr};
  alignas(kCacheLine) uint64_t w_{0};                 // producer-private
  std::byte* cur_{nullptr};
  std::byte* next_{nullptr};
  Stats st_;
  alignas(kCacheLine) std::atomic<uint64_t> pub_{0};  // polled by readers
  alignas(kCacheLine) std::atomic<bool> want_{false}; // LH <-> provisioner mailbox
  std::atomic<std::byte*> ready_{nullptr};
  std::vector<std::byte*> locked_;                    // provisioner-private
};

// ================================ QToTimeIdx =================================
// idx[gq][b] = first record index in gq with feed ts >= bucket_start(b), written by the
// owning LH when its feed time crosses a bucket boundary. Also carries each LH's watermark
// (latest published feed time) for cross-queue merging.
class TimeIndex {
 public:
  TimeIndex(uint64_t start_ns, uint64_t bucket_ns, uint32_t n_buckets, uint32_t n_lh, uint32_t q_per_lh)
      : start_(start_ns), bucket_(bucket_ns), nb_(n_buckets),
        idx_(std::make_unique<std::atomic<uint64_t>[]>(std::size_t{n_lh} * q_per_lh * n_buckets)),
        lh_(std::make_unique<LhSlot[]>(n_lh)) {}

  uint64_t start() const noexcept { return start_; }
  uint32_t n_buckets() const noexcept { return nb_; }
  uint64_t bucket_start(uint32_t b) const noexcept { return start_ + uint64_t{b} * bucket_; }
  uint32_t bucket_of(uint64_t ts) const noexcept {
    if (ts < start_) return 0;
    return static_cast<uint32_t>(std::min<uint64_t>((ts - start_) / bucket_, nb_ - 1));
  }
  void set(uint32_t gq, uint32_t b, uint64_t i) noexcept { idx_[std::size_t{gq} * nb_ + b].store(i, std::memory_order_relaxed); }
  uint64_t get(uint32_t gq, uint32_t b) const noexcept { return idx_[std::size_t{gq} * nb_ + b].load(std::memory_order_relaxed); }
  void publish(uint32_t lh, int32_t b) noexcept { lh_[lh].bucket.store(b, std::memory_order_release); }
  int32_t published(uint32_t lh) const noexcept { return lh_[lh].bucket.load(std::memory_order_acquire); }
  void set_watermark(uint32_t lh, uint64_t ts) noexcept { lh_[lh].wm.store(ts, std::memory_order_release); }
  uint64_t watermark(uint32_t lh) const noexcept { return lh_[lh].wm.load(std::memory_order_acquire); }

 private:
  struct alignas(kCacheLine) LhSlot { std::atomic<int32_t> bucket{-1}; std::atomic<uint64_t> wm{0}; };
  uint64_t start_, bucket_;
  uint32_t nb_;
  std::unique_ptr<std::atomic<uint64_t>[]> idx_;
  std::unique_ptr<LhSlot[]> lh_;
};

// ============================ Client output format ============================
struct OutQuote {        // 64 bytes, little-endian
  uint64_t feed_ns;      // original feed timestamp
  uint64_t rx_ns;        // receive timestamp
  int64_t  bid_px;
  int64_t  ask_px;       // x1e4
  uint32_t bid_sz;
  uint32_t ask_sz;
  char     sym[16];
  char     type, exch, bid_exch, ask_exch;
  uint8_t  proto;
  uint8_t  rsvd[3];
};
static_assert(sizeof(OutQuote) == 64);
using DecodeFn = bool (*)(const RecHdr&, const uint8_t* msg, OutQuote&) noexcept;

// ================================== Store ====================================
template <class L>
struct Store {
  using Q = MsgQueue<L>;
  struct Config {
    uint32_t    n_lh;
    uint32_t    q_per_lh;              // K (<= 64)
    std::size_t pool_chunks;
    std::size_t max_chunks_per_q;
    std::size_t max_symbols;
    uint64_t    session_start_ns;      // feed time domain
    uint64_t    bucket_ns;
    uint32_t    n_buckets;
    ChunkPool::Backing backing;
    std::string pool_path;             // Backing::File
    std::size_t keep_locked{3};        // chunks per queue kept mlocked (open + next + one back)
  };

  explicit Store(const Config& c)
      : cfg(c),
        pool(L::kBytes, c.pool_chunks, c.backing, c.pool_path),
        symmap(c.max_symbols),
        time(c.session_start_ns, c.bucket_ns, c.n_buckets, c.n_lh, c.q_per_lh),
        queues(std::make_unique<Q[]>(n_queues())) {
    decoders.fill(nullptr);
    for (std::size_t i = 0; i < n_queues(); ++i) queues[i].init(pool, c.max_chunks_per_q);
    provisioner = std::jthread([this](std::stop_token st) {
      while (!st.stop_requested()) {
        for (std::size_t i = 0; i < n_queues(); ++i) queues[i].provision_step(cfg.keep_locked);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    });
  }
  ~Store() { provisioner.request_stop(); provisioner.join(); }
  std::size_t n_queues() const noexcept { return std::size_t{cfg.n_lh} * cfg.q_per_lh; }
  uint32_t lh_of(uint32_t gq) const noexcept { return gq / cfg.q_per_lh; }

  Config cfg;
  ChunkPool pool;
  SymQMap symmap;
  TimeIndex time;
  std::unique_ptr<Q[]> queues;
  std::array<DecodeFn, 256> decoders;
  std::jthread provisioner;
};

// =============================== LineHandler =================================
// Proto contract (see toy_l1.hpp):
//   static constexpr uint8_t kProtoId;
//   template<class F> static void for_each_msg(const uint8_t* pkt, size_t len, F&& f);
//       f(const uint8_t* msg, uint32_t len, uint8_t type, const SymKey& key, uint64_t feed_ts)
//   static bool decode(const RecHdr&, const uint8_t* msg, OutQuote&) noexcept;
template <class Proto, class L>
class LineHandler {
  using Q = MsgQueue<L>;

 public:
  struct Stats { uint64_t pkts{0}, msgs{0}, new_symbols{0}, dropped{0}; };

  LineHandler(uint32_t lh_id, Store<L>& st)
      : st_(st), id_(lh_id), k_(st.cfg.q_per_lh), base_(lh_id * st.cfg.q_per_lh),
        q_(&st.queues[base_]), next_boundary_(st.time.start()) {
    if (k_ == 0 || k_ > 64) throw std::invalid_argument("q_per_lh must be 1..64");
    st.decoders[Proto::kProtoId] = &Proto::decode;
  }

  SymQMap::Entry pin(const SymKey& key, uint32_t k) { return st_.symmap.insert(key, base_ + k % k_); }

  void on_packet(const uint8_t* pkt, std::size_t len, uint64_t rx_ns, uint32_t aux = 0) noexcept {
    ++stats_.pkts;
    uint64_t dirty = 0;
    Proto::for_each_msg(pkt, len, [&](const uint8_t* m, uint32_t mlen, uint8_t type, const SymKey& key, uint64_t ts) {
      ts = std::max(ts, last_ts_);              // index space must be monotone per LH
      last_ts_ = ts;
      if (ts >= next_boundary_) [[unlikely]] roll(ts);
      const SymQMap::Entry e = resolve(key);
      const uint32_t k = e.gq - base_;
      const RecHdr h{static_cast<uint16_t>(mlen), type, Proto::kProtoId, aux, rx_ns};
      if (!q_[k].append(e.symid, ts, h, m)) [[unlikely]] ++stats_.dropped;
      dirty |= uint64_t{1} << k;
      ++stats_.msgs;
    });
    for (; dirty; dirty &= dirty - 1) q_[std::countr_zero(dirty)].publish();
    st_.time.set_watermark(id_, last_ts_);     // after the publishes: everything <= wm is readable
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
  [[gnu::cold, gnu::noinline]] void roll(uint64_t ts) noexcept {
    for (uint32_t k = 0; k < k_; ++k) q_[k].publish();  // idx values must be <= committed()
    TimeIndex& ti = st_.time;
    const uint32_t b = ti.bucket_of(ts);
    for (int64_t bb = cur_bucket_ + 1; bb <= int64_t{b}; ++bb)
      for (uint32_t k = 0; k < k_; ++k) ti.set(base_ + k, static_cast<uint32_t>(bb), q_[k].written());
    cur_bucket_ = b;
    ti.publish(id_, static_cast<int32_t>(b));
    next_boundary_ = (b + 1 < ti.n_buckets()) ? ti.bucket_start(b + 1) : UINT64_MAX;
  }

  Store<L>& st_;
  const uint32_t id_, k_, base_;
  Q* const q_;
  uint64_t next_boundary_;
  uint64_t last_ts_{0};
  int64_t cur_bucket_{-1};
  uint32_t rr_{0};
  Stats stats_;
};

// ================================ SymFilter ==================================
// Set of symids to match within one queue. <= 8 targets: AVX-512 compare per target,
// OR of masks (16 records per pass). More: bitset over symid.
class SymFilter {
 public:
  void add(uint32_t symid, std::size_t slots) {
    ids_.push_back(symid);
    if (ids_.size() > kSimdMax && bits_.empty()) bits_.assign((slots + 63) / 64, 0);
    if (!bits_.empty()) for (uint32_t id : ids_) bits_[id >> 6] |= uint64_t{1} << (id & 63);
  }
  [[gnu::always_inline]] unsigned match(const uint32_t* sym, __mmask16 lm) const noexcept {
    const __m512i v = _mm512_maskz_loadu_epi32(lm, sym);
    if (bits_.empty()) {
      unsigned m = 0;
      for (uint32_t id : ids_) m |= _mm512_mask_cmpeq_epi32_mask(lm, v, _mm512_set1_epi32(static_cast<int>(id)));
      return m;
    }
    unsigned m = 0;
    for (unsigned l = lm; l; l &= l - 1) {
      const unsigned j = static_cast<unsigned>(std::countr_zero(l));
      if (bits_[sym[j] >> 6] >> (sym[j] & 63) & 1) m |= 1u << j;
    }
    return m;
  }
 private:
  static constexpr std::size_t kSimdMax = 8;
  std::vector<uint32_t> ids_;
  std::vector<uint64_t> bits_;
};

// =============================== QueueCursor =================================
// Read-side state for one queue within a client session: start-point search, seed
// (last record of a symbol before the start), and forward scan to the next match.
template <class L>
class QueueCursor {
  using Q = MsgQueue<L>;

 public:
  QueueCursor(const Store<L>& st, uint32_t gq) : st_(st), q_(st.queues[gq]), gq_(gq), lh_(st.lh_of(gq)) {}
  void add_symbol(uint32_t symid) { filter_.add(symid, st_.symmap.slots()); }
  uint32_t gq() const noexcept { return gq_; }
  uint32_t lh() const noexcept { return lh_; }
  const Q& queue() const noexcept { return q_; }

  // First index with ts >= t. QToTimeIdx narrows to one bucket; ts sidecar makes it exact.
  uint64_t find_start(uint64_t t) const noexcept {
    const TimeIndex& ti = st_.time;
    const int32_t pb = ti.published(lh_);  // before committed(): idx[<=pb] <= w
    const uint64_t w = q_.committed();
    uint64_t lo = 0, hi = w;
    if (pb >= 0 && t >= ti.start()) {
      const uint32_t b = ti.bucket_of(t);
      if (static_cast<int32_t>(b) <= pb) { lo = ti.get(gq_, b); if (static_cast<int32_t>(b) < pb) hi = ti.get(gq_, b + 1); }
      else lo = ti.get(gq_, static_cast<uint32_t>(pb));
    }
    hi = std::min(hi, w); lo = std::min(lo, hi);
    while (lo < hi) { const uint64_t mid = lo + (hi - lo) / 2; if (q_.ts_at(mid) < t) lo = mid + 1; else hi = mid; }
    return lo;
  }
  void seek(uint64_t i) noexcept { pos_ = i; next_.reset(); }
  uint64_t pos() const noexcept { return pos_; }

  // Last record of one symbol strictly before `end`, scanning at most max_scan records back.
  std::optional<uint64_t> find_prior(uint32_t symid, uint64_t end, uint64_t max_scan) const noexcept {
    const __m512i tgt = _mm512_set1_epi32(static_cast<int>(symid));
    const uint64_t stop = end > max_scan ? end - max_scan : 0;
    uint64_t i = end;
    while (i > stop) {
      const uint64_t c = (i - 1) >> L::kShift, cbase = c << L::kShift;
      const uint64_t lo = std::max(cbase, stop);
      const uint32_t* sym = Q::sym_of(q_.chunk(c));
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

  // Next matching record at index >= pos_, below `limit` (a committed() value). Cached.
  std::optional<uint64_t> peek(uint64_t limit) noexcept {
    if (next_ && *next_ < limit) return next_;
    uint64_t i = pos_;
    while (i < limit) {
      const uint64_t c = i >> L::kShift, cbase = c << L::kShift;
      const uint64_t cend = std::min<uint64_t>(limit, cbase + L::kCap);
      const uint32_t* sym = Q::sym_of(q_.chunk(c));
      for (; i < cend; i += 16) {
        const auto s = static_cast<uint32_t>(i - cbase);
        const auto n = static_cast<unsigned>(std::min<uint64_t>(16, cend - i));
        _mm_prefetch(reinterpret_cast<const char*>(sym + s + 256), _MM_HINT_T0);
        if (const unsigned m = filter_.match(sym + s, static_cast<__mmask16>(_bzhi_u32(0xFFFFu, n)))) {
          next_ = cbase + s + static_cast<unsigned>(std::countr_zero(m));
          pos_  = *next_;
          return next_;
        }
      }
      i = cend;  // the 16-wide step may overshoot the chunk end
    }
    pos_ = limit;
    next_.reset();
    return std::nullopt;
  }
  void consume() noexcept { pos_ = *next_ + 1; next_.reset(); }

  uint64_t ts_at(uint64_t i) const noexcept { return q_.ts_at(i); }
  template <class F> void visit(uint64_t i, F&& f) const {
    const std::byte* ch = q_.chunk(i >> L::kShift);
    const std::byte* r = Q::rec_of(ch, i & L::kMask);
    f(*reinterpret_cast<const RecHdr*>(r), reinterpret_cast<const uint8_t*>(r + kRecHdr), Q::ts_of(ch)[i & L::kMask]);
  }

 private:
  const Store<L>& st_;
  const Q& q_;
  uint32_t gq_, lh_;
  SymFilter filter_;
  uint64_t pos_{0};
  std::optional<uint64_t> next_;
};

}  // namespace rawmd
