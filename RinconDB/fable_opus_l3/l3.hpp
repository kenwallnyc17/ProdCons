#pragma once
// l3.hpp — Level-3 (order-by-order) analog of the L1 store.
//
// Ingest: raw order messages are stored per symbol in the same append-only queues. The
// line handler decodes only what it needs to route a message to its symbol's queue:
//   * feeds with an instrument/locate index in every message (ITCH, TSX): a direct
//     locate -> {gq, symid} table, filled from directory messages;
//   * feeds that carry the symbol only on Add (PITCH-style): an order-id route table
//     (OrderRoute) maintained from Add/Reduce/End/Replace/Modify.
//
// Replay (l3_server.hpp): a client asks for N symbols from feed time T. Each symbol's book
// is built from session start; once it has consumed every message before T the session
// publishes a top-N snapshot, then the top N after every later message that changes it,
// in feed-time order across symbols, and keeps doing so live.
//
// Book: open-addressing order table plus one blocked sorted ladder per side (64-level
// blocks, SoA keys for AVX-512 search). Top N is a prefix walk; a level insert or erase
// moves at most one block.

#include "rawmd.hpp"

#include <new>
#include <type_traits>

namespace rawmd::l3 {

// ============================ Protocol contract ==============================
// struct Proto {
//   static constexpr uint8_t kProtoId;
//   static constexpr bool    kHasLocate;   // every message carries a locate/instrument index
//   static constexpr int8_t  kPxExp;       // price = int * 10^kPxExp
//   template<class F> static void for_each_msg(const uint8_t* pkt, size_t len, F&& f);
//       f(const uint8_t* msg, uint32_t len, uint8_t type, uint64_t feed_ts, const MsgRef& ref)
//   static bool decode(const RecHdr&, const uint8_t* msg, BookEvent&) noexcept;
// };

enum class Op : uint8_t {
  Skip,     // no symbol (system event, heartbeat): not stored, advances the LH clock
  Dir,      // directory / instrument definition: key valid; registers the locate
  Sym,      // symbol-bearing, no order effect (trading action, imbalance, trade report)
  Add,      // key (when !kHasLocate), order_id, qty
  Ref,      // references an order, no qty change
  Reduce,   // executed or cancelled qty: order_id, qty
  End,      // order removed: order_id
  Replace,  // order_id -> new_order_id, qty
  Modify,   // same order_id, qty replaced
};

struct MsgRef {
  SymKey   key{};
  uint64_t order_id{0};
  uint64_t new_order_id{0};
  uint32_t qty{0};
  uint16_t locate{0};
  Op       op{Op::Skip};
};

enum class Ev : uint8_t { None, Add, Reduce, Delete, Replace, Modify, Clear };

struct BookEvent {
  Ev       ev{Ev::None};
  uint8_t  side{0};        // 'B' or 'S' (Add only; carried by the order afterwards)
  uint32_t qty{0};         // Add/Replace/Modify: order qty; Reduce: qty removed
  int64_t  px{0};          // Add/Replace/Modify
  uint64_t order_id{0};
  uint64_t new_order_id{0};
};

// ================================ HugeMem ====================================
// Zeroed memory for hot-path tables: 2 MB pages when available (hugetlbfs reservation,
// else THP) and touched at construction, so lookups never page-fault and rarely TLB-miss.
class HugeMem {
 public:
  explicit HugeMem(std::size_t bytes) : bytes_(round_up(bytes, kHugePage)) {
    void* p = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
    if (p == MAP_FAILED) {
      p = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (p == MAP_FAILED) throw std::bad_alloc();
      ::madvise(p, bytes_, MADV_HUGEPAGE);
      std::memset(p, 0, bytes_);
    }
    p_ = p;
  }
  ~HugeMem() { ::munmap(p_, bytes_); }
  HugeMem(const HugeMem&) = delete;
  HugeMem& operator=(const HugeMem&) = delete;
  void* get() const noexcept { return p_; }

 private:
  std::size_t bytes_;
  void* p_{nullptr};
};

// ================================ OpenTable ==================================
// u64 -> Val, linear probing without wrap, grows at 75% load. Probes walk aligned 64-byte
// lines of 8 keys (one AVX-512 compare per line; in the first line, lanes before the home
// slot are masked out of the empty test), so nothing is ever written past cap + kPad - 1
// and the 8 keys after that are a permanent zero sentinel for backward-shift delete. A
// cluster that reaches the end of the padding triggers a grow. Key 0 is the empty marker;
// keys are stored xor kSalt so an id of 0 works (an id equal to kSalt is the one value
// that cannot be stored).
template <class Val>
class OpenTable {
  static_assert(std::is_trivially_copyable_v<Val>);

 public:
  static constexpr std::size_t npos = ~std::size_t{0};
  struct Ins { std::size_t i; bool fresh; };   // fresh == false: key was already present at i

  explicit OpenTable(unsigned log2cap) {
    if (log2cap < 3 || log2cap > 40) throw std::invalid_argument("OpenTable size");
    alloc(log2cap);
  }

  [[gnu::always_inline]] std::size_t find(uint64_t k) const noexcept {
    const __m512i bk = _mm512_set1_epi64(static_cast<long long>(k ^ kSalt));
    const std::size_t h = home(k);
    auto valid = static_cast<__mmask8>(0xFFu << (h & 7));
    for (std::size_t i = h & ~std::size_t{7}; i < cap_ + kPad; i += 8, valid = 0xFF) {
      const __m512i v = _mm512_load_si512(keys_.get() + i);
      if (const unsigned m = _mm512_cmpeq_epi64_mask(v, bk)) return i + static_cast<unsigned>(std::countr_zero(m));
      if (_mm512_mask_cmpeq_epi64_mask(valid, v, _mm512_setzero_si512())) return npos;
    }
    return npos;
  }

  [[gnu::always_inline]] Ins insert(uint64_t k, const Val& v) {
    if (n_ * 4 >= cap_ * 3) [[unlikely]] grow();
    if (const Ins r = try_insert(k, v); r.i != npos) [[likely]] return r;
    return insert_slow(k, v);
  }

  void erase(std::size_t i) noexcept {
    for (std::size_t j = i + 1;; ++j) {             // stops at the sentinel at the latest
      const uint64_t kj = keys_[j];
      if (kj == 0) break;
      if (home(kj ^ kSalt) <= i) { keys_[i] = kj; vals_[i] = vals_[j]; i = j; }
    }
    keys_[i] = 0;
    --n_;
  }

  Val& val(std::size_t i) noexcept { return vals_[i]; }
  const Val& val(std::size_t i) const noexcept { return vals_[i]; }
  void clear() noexcept { std::memset(keys_.get(), 0, slots() * sizeof(uint64_t)); n_ = 0; }
  std::size_t size() const noexcept { return n_; }
  std::size_t capacity() const noexcept { return cap_; }
  std::size_t home(uint64_t k) const noexcept { return static_cast<std::size_t>((k * kFib) >> shift_); }

 private:
  static constexpr uint64_t kFib = 0x9E3779B97F4A7C15ull, kSalt = kFib;
  static constexpr std::size_t kPad = 256;
  struct Free { void operator()(uint64_t* p) const noexcept { ::operator delete[](p, std::align_val_t{64}); } };

  std::size_t slots() const noexcept { return cap_ + kPad + 8; }

  [[gnu::noinline, gnu::cold]] Ins insert_slow(uint64_t k, const Val& v) {
    for (;;) {                                      // the cluster ran into the end of the padding
      grow();
      if (const Ins r = try_insert(k, v); r.i != npos) return r;
    }
  }
  [[gnu::always_inline]] Ins try_insert(uint64_t k, const Val& v) noexcept {
    const uint64_t sk = k ^ kSalt;
    const __m512i bk = _mm512_set1_epi64(static_cast<long long>(sk));
    const std::size_t h = home(k);
    auto valid = static_cast<__mmask8>(0xFFu << (h & 7));
    for (std::size_t i = h & ~std::size_t{7}; i < cap_ + kPad; i += 8, valid = 0xFF) {
      const __m512i kv = _mm512_load_si512(keys_.get() + i);
      if (const unsigned m = _mm512_cmpeq_epi64_mask(kv, bk)) return {i + static_cast<unsigned>(std::countr_zero(m)), false};
      if (const unsigned e = _mm512_mask_cmpeq_epi64_mask(valid, kv, _mm512_setzero_si512())) {
        i += static_cast<unsigned>(std::countr_zero(e));
        keys_[i] = sk;
        vals_[i] = v;
        ++n_;
        return {i, true};
      }
    }
    return {npos, false};
  }
  void alloc(unsigned log2cap) {
    cap_ = std::size_t{1} << log2cap;
    shift_ = 64 - log2cap;
    keys_.reset(static_cast<uint64_t*>(::operator new[](slots() * sizeof(uint64_t), std::align_val_t{64})));
    std::memset(keys_.get(), 0, slots() * sizeof(uint64_t));
    vals_ = std::make_unique_for_overwrite<Val[]>(slots());
    n_ = 0;
  }
  [[gnu::noinline, gnu::cold]] void grow() {
    auto ok = std::move(keys_);
    auto ov = std::move(vals_);
    const std::size_t used = cap_ + kPad;
    alloc(64 - shift_ + 1);
    for (std::size_t i = 0; i < used; ++i)
      if (ok[i]) insert(ok[i] ^ kSalt, ov[i]);
  }

  std::unique_ptr<uint64_t[], Free> keys_;
  std::unique_ptr<Val[]> vals_;
  std::size_t cap_{0}, n_{0};
  unsigned shift_{64};
};

// ================================ OrderRoute =================================
// LH-side order id -> (local queue, symid, remaining qty), for feeds whose messages after
// the Add carry only the order id. On the hot path for every such message, so: 16-byte
// entries, four per cache line; a probe starts at the aligned line holding the home slot
// and tests the four keys with one AVX-512 compare, so a lookup is one line in the common
// case. Linear probing without wrap, backward-shift delete, fixed capacity on 2 MB pages,
// prefaulted. kPad is large because a full tail means a dropped message, not a grow.
// Size it for at least 2x the peak number of live orders on the line handler.
class OrderRoute {
 public:
  struct alignas(16) Ent { uint64_t key; uint32_t ks; uint32_t rem; };   // ks = local queue << kSymBits | symid
  static_assert(sizeof(Ent) == 16);
  struct Ins { std::size_t i; bool fresh; };
  static constexpr std::size_t npos = ~std::size_t{0};
  static constexpr unsigned kSymBits = 26;
  static constexpr uint32_t kSymMask = (1u << kSymBits) - 1;

  explicit OrderRoute(unsigned log2cap)
      : cap_(std::size_t{1} << log2cap), shift_(64 - log2cap),
        mem_((cap_ + kPad + 4) * sizeof(Ent)), e_(static_cast<Ent*>(mem_.get())) {
    if (log2cap < 2 || log2cap > 40) throw std::invalid_argument("OrderRoute size");
  }

  [[gnu::always_inline]] std::size_t find(uint64_t k) const noexcept {
    const __m512i bk = _mm512_set1_epi64(static_cast<long long>(k ^ kSalt));
    const std::size_t h = home(k);
    auto valid = static_cast<__mmask8>(0x55u << (2 * (h & 3)));          // key lanes at or after home
    for (std::size_t i = h & ~std::size_t{3}; i < cap_ + kPad; i += 4, valid = 0x55) {
      const __m512i v = _mm512_load_si512(e_ + i);
      if (const unsigned m = _mm512_mask_cmpeq_epi64_mask(0x55, v, bk)) return i + (static_cast<unsigned>(std::countr_zero(m)) >> 1);
      if (_mm512_mask_cmpeq_epi64_mask(valid, v, _mm512_setzero_si512())) return npos;
    }
    return npos;
  }

  // npos when the table is full (counted by the caller; the message cannot be routed).
  [[gnu::always_inline]] Ins insert(uint64_t k, uint32_t ks, uint32_t rem) noexcept {
    const uint64_t sk = k ^ kSalt;
    const __m512i bk = _mm512_set1_epi64(static_cast<long long>(sk));
    const std::size_t h = home(k);
    auto valid = static_cast<__mmask8>(0x55u << (2 * (h & 3)));
    for (std::size_t i = h & ~std::size_t{3}; i < cap_ + kPad; i += 4, valid = 0x55) {
      const __m512i v = _mm512_load_si512(e_ + i);
      if (const unsigned m = _mm512_mask_cmpeq_epi64_mask(0x55, v, bk)) return {i + (static_cast<unsigned>(std::countr_zero(m)) >> 1), false};
      if (const unsigned z = _mm512_mask_cmpeq_epi64_mask(valid, v, _mm512_setzero_si512())) {
        const std::size_t j = i + (static_cast<unsigned>(std::countr_zero(z)) >> 1);
        e_[j] = {sk, ks, rem};
        ++n_;
        return {j, true};
      }
    }
    return {npos, false};
  }

  void erase(std::size_t i) noexcept {
    for (std::size_t j = i + 1;; ++j) {             // e_[cap_ + kPad ..] is never written: sentinel
      const uint64_t kj = e_[j].key;
      if (kj == 0) break;
      if (home(kj ^ kSalt) <= i) { e_[i] = e_[j]; i = j; }
    }
    e_[i].key = 0;
    --n_;
  }

  Ent& at(std::size_t i) noexcept { return e_[i]; }
  std::size_t size() const noexcept { return n_; }
  std::size_t capacity() const noexcept { return cap_; }
  std::size_t home(uint64_t k) const noexcept { return static_cast<std::size_t>((k * kFib) >> shift_); }

 private:
  static constexpr uint64_t kFib = 0x9E3779B97F4A7C15ull, kSalt = kFib;
  static constexpr std::size_t kPad = 1024;

  std::size_t cap_;
  unsigned shift_;
  HugeMem mem_;
  Ent* e_;
  std::size_t n_{0};
};

// ================================ LevelSide ==================================
// One side of an aggregate book: levels sorted best-first (key = px for asks, -px for
// bids), in blocks of 64. Each block is SoA (keys, qtys, counts) so the in-block search
// is a masked compare + popcount, and the top N is read straight out of the first block(s).
struct Level { int64_t px; uint32_t qty; uint32_t cnt; bool operator==(const Level&) const = default; };
static_assert(sizeof(Level) == 16);

class LevelSide {
 public:
  static constexpr uint32_t kB = 64;

  explicit LevelSide(bool bid) : bid_(bid) {}

  // Each returns the rank (0 = best) of the level touched, for "did the top N change".
  uint32_t add(int64_t px, uint32_t qty, uint32_t rank_limit) {
    const int64_t key = bid_ ? -px : px;
    Pos p = locate(key, rank_limit);
    if (p.found) { Block& B = *blocks_[p.b]; B.qty[p.p] += qty; B.cnt[p.p] += 1; }
    else insert_at(p, key, qty);
    return p.rank;
  }
  uint32_t reduce(int64_t px, uint32_t qty, bool order_gone, uint32_t rank_limit) {
    const int64_t key = bid_ ? -px : px;
    const Pos p = locate(key, rank_limit);
    if (!p.found) [[unlikely]] return UINT32_MAX;  // level missing: inconsistent feed, ignore
    Block& B = *blocks_[p.b];
    B.qty[p.p] = B.qty[p.p] > qty ? B.qty[p.p] - qty : 0;
    if (order_gone && --B.cnt[p.p] == 0) erase_at(p);
    return p.rank;
  }
  uint32_t top(uint32_t n, Level* out) const noexcept {
    uint32_t k = 0;
    for (const auto& bp : blocks_) {
      const Block& B = *bp;
      for (uint32_t i = 0; i < B.n && k < n; ++i, ++k) out[k] = {bid_ ? -B.key[i] : B.key[i], B.qty[i], B.cnt[i]};
      if (k >= n) break;
    }
    return k;
  }
  std::size_t levels() const noexcept { std::size_t s = 0; for (const auto& b : blocks_) s += b->n; return s; }
  void clear() noexcept { for (auto& b : blocks_) free_.push_back(std::move(b)); blocks_.clear(); lo_.clear(); }

 private:
  struct alignas(64) Block { uint32_t n; uint32_t pad[15]; int64_t key[kB]; uint32_t qty[kB]; uint32_t cnt[kB]; };
  struct Pos { uint32_t b, p, rank; bool found; };

  Pos locate(int64_t key, uint32_t rank_limit) const noexcept {
    const std::size_t nb = blocks_.size();
    if (nb == 0) return {0, 0, 0, false};
    const __m512i bk = _mm512_set1_epi64(static_cast<long long>(key));
    // Both counts are branch-free: a full masked scan pipelines better than an early exit
    // whose position changes every call.
    uint32_t c = 0;                                   // blocks whose first key <= key
    for (std::size_t i = 0; i < nb; i += 8) {
      const auto lm = static_cast<__mmask8>(_bzhi_u32(0xFFu, static_cast<unsigned>(std::min<std::size_t>(8, nb - i))));
      c += static_cast<uint32_t>(std::popcount(_mm512_mask_cmple_epi64_mask(lm, _mm512_maskz_loadu_epi64(lm, lo_.data() + i), bk)));
    }
    const uint32_t b = c ? c - 1 : 0;
    const Block& B = *blocks_[b];
    uint32_t p = 0;                                   // keys in block < key
    for (uint32_t i = 0; i < B.n; i += 8) {
      const auto lm = static_cast<__mmask8>(_bzhi_u32(0xFFu, std::min<uint32_t>(8, B.n - i)));
      p += static_cast<uint32_t>(std::popcount(_mm512_mask_cmplt_epi64_mask(lm, _mm512_loadu_si512(B.key + i), bk)));
    }
    uint32_t rank = p;
    for (uint32_t i = 0; i < b && rank <= rank_limit; ++i) rank += blocks_[i]->n;
    return {b, p, rank, p < B.n && B.key[p] == key};
  }
  Block* take() {
    if (free_.empty()) return new Block{};
    Block* b = free_.back().release(); free_.pop_back(); b->n = 0; return b;
  }
  void insert_at(Pos p, int64_t key, uint32_t qty) {
    if (blocks_.empty()) { blocks_.emplace_back(take()); lo_.push_back(key); }
    Block* B = blocks_[p.b].get();
    if (B->n == kB) {                                 // split: upper half moves to a new block
      Block* N = take();
      N->n = kB / 2;
      std::memcpy(N->key, B->key + kB / 2, sizeof(int64_t) * (kB / 2));
      std::memcpy(N->qty, B->qty + kB / 2, sizeof(uint32_t) * (kB / 2));
      std::memcpy(N->cnt, B->cnt + kB / 2, sizeof(uint32_t) * (kB / 2));
      B->n = kB / 2;
      blocks_.emplace(blocks_.begin() + p.b + 1, N);
      lo_.insert(lo_.begin() + p.b + 1, N->key[0]);
      if (p.p > kB / 2) { ++p.b; p.p -= kB / 2; B = N; }
    }
    const uint32_t tail = B->n - p.p;
    std::memmove(B->key + p.p + 1, B->key + p.p, sizeof(int64_t) * tail);
    std::memmove(B->qty + p.p + 1, B->qty + p.p, sizeof(uint32_t) * tail);
    std::memmove(B->cnt + p.p + 1, B->cnt + p.p, sizeof(uint32_t) * tail);
    B->key[p.p] = key; B->qty[p.p] = qty; B->cnt[p.p] = 1;
    ++B->n;
    if (p.p == 0) lo_[p.b] = key;
  }
  void erase_at(Pos p) {
    Block* B = blocks_[p.b].get();
    const uint32_t tail = B->n - p.p - 1;
    std::memmove(B->key + p.p, B->key + p.p + 1, sizeof(int64_t) * tail);
    std::memmove(B->qty + p.p, B->qty + p.p + 1, sizeof(uint32_t) * tail);
    std::memmove(B->cnt + p.p, B->cnt + p.p + 1, sizeof(uint32_t) * tail);
    if (--B->n == 0) {
      free_.push_back(std::move(blocks_[p.b]));
      blocks_.erase(blocks_.begin() + p.b);
      lo_.erase(lo_.begin() + p.b);
      return;
    }
    if (p.p == 0) lo_[p.b] = B->key[0];
    if (B->n <= kB / 4 && p.b + 1 < blocks_.size() && B->n + blocks_[p.b + 1]->n <= 3 * kB / 4) {
      Block* N = blocks_[p.b + 1].get();              // merge the next block into this one
      std::memcpy(B->key + B->n, N->key, sizeof(int64_t) * N->n);
      std::memcpy(B->qty + B->n, N->qty, sizeof(uint32_t) * N->n);
      std::memcpy(B->cnt + B->n, N->cnt, sizeof(uint32_t) * N->n);
      B->n += N->n;
      free_.push_back(std::move(blocks_[p.b + 1]));
      blocks_.erase(blocks_.begin() + p.b + 1);
      lo_.erase(lo_.begin() + p.b + 1);
    }
  }

  bool bid_;
  std::vector<std::unique_ptr<Block>> blocks_;
  std::vector<int64_t> lo_;                           // blocks_[i]->key[0]
  std::vector<std::unique_ptr<Block>> free_;
};

// =================================== Book ====================================
class Book {
 public:
  struct Stats { uint64_t applied{0}, unknown_order{0}, dup_order{0}; };

  explicit Book(uint32_t depth, unsigned log2orders = 12) : depth_(depth), orders_(log2orders), bid_(true), ask_(false) {}

  // Applies one event; true if the top `depth` levels of either side may have changed.
  bool apply(const BookEvent& e) {
    ++st_.applied;
    switch (e.ev) {
      case Ev::Add: {
        if (!orders_.insert(e.order_id, {e.px, e.qty, e.side}).fresh) [[unlikely]] { ++st_.dup_order; return false; }
        return side(e.side).add(e.px, e.qty, depth_) < depth_;
      }
      case Ev::Reduce: {
        const std::size_t i = orders_.find(e.order_id);
        if (i == Tbl::npos) [[unlikely]] { ++st_.unknown_order; return false; }
        Order& o = orders_.val(i);
        const uint32_t q = std::min(e.qty, o.qty);
        o.qty -= q;
        const bool gone = o.qty == 0;
        const uint32_t r = side(o.side).reduce(o.px, q, gone, depth_);
        if (gone) orders_.erase(i);
        return r < depth_;
      }
      case Ev::Delete: {
        const std::size_t i = orders_.find(e.order_id);
        if (i == Tbl::npos) [[unlikely]] { ++st_.unknown_order; return false; }
        const Order o = orders_.val(i);
        orders_.erase(i);
        return side(o.side).reduce(o.px, o.qty, true, depth_) < depth_;
      }
      case Ev::Replace: {
        const std::size_t i = orders_.find(e.order_id);
        if (i == Tbl::npos) [[unlikely]] { ++st_.unknown_order; return false; }
        const Order o = orders_.val(i);
        orders_.erase(i);
        const uint32_t r1 = side(o.side).reduce(o.px, o.qty, true, depth_);
        if (!orders_.insert(e.new_order_id, {e.px, e.qty, o.side}).fresh) [[unlikely]] { ++st_.dup_order; return r1 < depth_; }
        const uint32_t r2 = side(o.side).add(e.px, e.qty, depth_);
        return std::min(r1, r2) < depth_;
      }
      case Ev::Modify: {
        const std::size_t i = orders_.find(e.order_id);
        if (i == Tbl::npos) [[unlikely]] { ++st_.unknown_order; return false; }
        Order& o = orders_.val(i);
        const uint32_t r1 = side(o.side).reduce(o.px, o.qty, true, depth_);
        o.px = e.px; o.qty = e.qty;
        const uint32_t r2 = side(o.side).add(e.px, e.qty, depth_);
        return std::min(r1, r2) < depth_;
      }
      case Ev::Clear: orders_.clear(); bid_.clear(); ask_.clear(); return true;
      case Ev::None: return false;
    }
    return false;
  }
  uint32_t top_bids(Level* out) const noexcept { return bid_.top(depth_, out); }
  uint32_t top_asks(Level* out) const noexcept { return ask_.top(depth_, out); }
  uint32_t depth() const noexcept { return depth_; }
  std::size_t orders() const noexcept { return orders_.size(); }
  std::size_t bid_levels() const noexcept { return bid_.levels(); }
  std::size_t ask_levels() const noexcept { return ask_.levels(); }
  // False once an event referenced an unknown order or reused a live order id: the input
  // is not self-consistent (e.g. the store started after the open without a snapshot).
  bool consistent() const noexcept { return st_.unknown_order == 0 && st_.dup_order == 0; }
  const Stats& stats() const noexcept { return st_; }

 private:
  struct Order { int64_t px; uint32_t qty; uint8_t side; uint8_t pad[3]{}; };
  using Tbl = OpenTable<Order>;
  LevelSide& side(uint8_t s) noexcept { return s == 'B' ? bid_ : ask_; }

  uint32_t depth_;
  Tbl orders_;
  LevelSide bid_, ask_;
  Stats st_;
};

// ============================== LineHandler3 =================================
template <class Proto, class L>
class LineHandler3 {
  using Q = MsgQueue<L>;

 public:
  struct Stats {
    uint64_t pkts{0}, msgs{0}, new_symbols{0}, dropped{0}, oversize{0};
    uint64_t unknown_locate{0}, unknown_order{0}, dup_order{0}, route_full{0};
  };

  // log2_orders: OrderRoute size for feeds without a locate (ignored otherwise). Size it for
  // at least 2x the peak number of live orders on this line handler.
  LineHandler3(uint32_t lh_id, Store<L>& st, unsigned log2_orders = 22) : core_(lh_id, st) {
    if constexpr (Proto::kHasLocate) {
      loc_ = std::make_unique<SymQMap::Entry[]>(65536);
      for (std::size_t i = 0; i < 65536; ++i) loc_[i] = {kNoGq, 0};
    } else {
      if (st.symmap.slots() > (std::size_t{1} << OrderRoute::kSymBits)) throw std::invalid_argument("symbol table too large for OrderRoute");
      route_ = std::make_unique<OrderRoute>(log2_orders);
    }
  }

  void advance_time(uint64_t ts) noexcept { core_.advance_time(ts); }

  void on_packet(const uint8_t* pkt, std::size_t len, uint64_t rx_ns, uint32_t aux = 0) noexcept {
    ++stats_.pkts;
    uint64_t dirty = 0;
    Q* const q = core_.queues();
    Proto::for_each_msg(pkt, len, [&](const uint8_t* m, uint32_t mlen, uint8_t type, uint64_t ts, const MsgRef& ref) {
      ts = core_.clock(ts);
      if (ref.op == Op::Skip) return;             // clock advanced; watermark moves at commit
      SymQMap::Entry e;
      if (!resolve(ref, e)) [[unlikely]] return;
      const uint32_t k = e.gq - core_.base();
      if (mlen > L::kMaxMsg) [[unlikely]] { ++stats_.oversize; return; }
      const RecHdr h{static_cast<uint16_t>(mlen), type, Proto::kProtoId, aux, rx_ns};
      if (!q[k].append(e.symid, ts, h, m)) [[unlikely]] ++stats_.dropped;
      dirty |= uint64_t{1} << k;
      ++stats_.msgs;
    });
    core_.commit(dirty);
  }
  const Stats& stats() noexcept { stats_.new_symbols = core_.new_symbols(); return stats_; }

 private:
  static constexpr uint32_t kNoGq = UINT32_MAX;

  uint32_t pack(SymQMap::Entry e) const noexcept { return (e.gq - core_.base()) << OrderRoute::kSymBits | e.symid; }
  SymQMap::Entry unpack(uint32_t ks) const noexcept { return {core_.base() + (ks >> OrderRoute::kSymBits), ks & OrderRoute::kSymMask}; }

  [[gnu::always_inline]] bool resolve(const MsgRef& r, SymQMap::Entry& e) noexcept {
    if constexpr (Proto::kHasLocate) {
      SymQMap::Entry& slot = loc_[r.locate];
      if (slot.gq != kNoGq) [[likely]] { e = slot; return true; }
      if (r.op == Op::Dir) { slot = core_.assign(r.key); e = slot; return true; }
      ++stats_.unknown_locate;
      return false;
    } else {
      switch (r.op) {
        case Op::Dir: case Op::Sym: e = core_.resolve(r.key); return true;
        case Op::Add: {
          e = core_.resolve(r.key);
          const auto [i, fresh] = route_->insert(r.order_id, pack(e), r.qty);
          if (i == OrderRoute::npos) [[unlikely]] ++stats_.route_full;
          else if (!fresh) [[unlikely]] { ++stats_.dup_order; route_->at(i).ks = pack(e); route_->at(i).rem = r.qty; }   // id reused: follow the new order
          return true;
        }
        default: break;
      }
      const std::size_t i = route_->find(r.order_id);
      if (i == OrderRoute::npos) [[unlikely]] { ++stats_.unknown_order; return false; }
      OrderRoute::Ent& v = route_->at(i);
      e = unpack(v.ks);
      switch (r.op) {
        case Op::Reduce: v.rem = v.rem > r.qty ? v.rem - r.qty : 0; if (v.rem == 0) route_->erase(i); break;
        case Op::End:    route_->erase(i); break;
        case Op::Modify: v.rem = r.qty; break;
        case Op::Replace: {
          const uint32_t ks = v.ks;
          route_->erase(i);
          const auto [j, fresh] = route_->insert(r.new_order_id, ks, r.qty);
          if (j == OrderRoute::npos) [[unlikely]] ++stats_.route_full;
          else if (!fresh) [[unlikely]] { ++stats_.dup_order; route_->at(j).ks = ks; route_->at(j).rem = r.qty; }
          break;
        }
        default: break;
      }
      return true;
    }
  }

  LhCore<L> core_;
  std::unique_ptr<SymQMap::Entry[]> loc_;   // kHasLocate
  std::unique_ptr<OrderRoute> route_;       // !kHasLocate
  Stats stats_;
};

// ============================ Client output format ===========================
struct OutBookHdr {        // 48 bytes, followed by nbid Level then nask Level
  uint64_t feed_ns;        // 'S': snapshot time (book after every message before it); 'U': the message's feed time
  uint64_t rx_ns;
  char     sym[16];
  uint32_t seq;            // per-symbol update sequence; 0 for the snapshot
  uint16_t nbid, nask;
  uint8_t  kind;           // 'S' snapshot, 'U' update
  int8_t   px_exp;
  uint8_t  trigger;        // feed message type that caused an update
  uint8_t  flags;          // kFlagInconsistent: see Book::consistent()
  uint8_t  rsvd[4];
};
static_assert(sizeof(OutBookHdr) == 48);
inline constexpr uint8_t  kFlagInconsistent = 1;
inline constexpr uint32_t kMaxDepth = 64;

}  // namespace rawmd::l3
