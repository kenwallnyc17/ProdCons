// mdsys.hpp — HFT market data system:
//
//  [1] OrderTable: order-id hash -> BUCKET OF 2 CACHE LINES (128B) holding 4
//      orders SoA. keys[4] are 32 contiguous bytes, so ONE AVX2 cmpeq_epi64
//      scans the whole bucket. key==0 means deleted/free and is reused.
//      Black swan: a full bucket chains additional preallocated 128B buckets
//      from a pool (same layout -> same scan code); an overflow bucket that
//      empties is unlinked and returned to the pool.
//
//  [2] SymbolState[symst_idx], symst_idx < 2^14: per-symbol config (tick,
//      round lot, market state) plus bid and ask books, and the state the
//      join and cancel tactics share (TacticShared: per side, the live join
//      signal, the executions at or through its price since, and the most
//      aggressive execution prices since, by resting side).
//
//  [3] LadderBook: paged ladder. Price -> tick index -> (dir slot, page slot)
//      by shift/mask; pages of 512 PrcLevelGeneric bound lazily from a shared
//      preallocated pool and recycled when empty. Occupancy bitmaps (per-word
//      + summary) give tzcnt/lzcnt inside-recovery and sorted traversal.
//      Prices outside the window / off the tick grid go to a small per-book
//      overflow store (fail-closed: correct best/roundlot, never reject,
//      never evict). Order.ordq_idx encodes the level handle: bit31 set ->
//      overflow slot, else the tick-relative index; E/C/X/D therefore never
//      redo the price math — only Add pays the division.
//
//  [4] CancelTactic: resting-order cancel decisions, re-checked on every
//      venue book update of a tracked symbol (add / delete / modify /
//      cancel / replace / execute); a symbol with no active orders (and not
//      pinned) costs one bit test. (a) >= N orders and >= K x our size
//      ahead in the level's queue (N=5, K=3): at placement, and again when
//      the shares ahead can have grown against our size; (b) our leading
//      order at a price is at the front of the queue or has only odd lots
//      ahead of it, and more than X% (50) of the round-lot orders that were
//      ahead of it were cancelled (not filled): every order of ours at that
//      price is cancelled; (c) too deep vs the far NBBO
//      touch, threshold = max(1.5*hist spread (fixed per symbol), spread,
//      5 ticks, sigma*sqrt(min(5 min, 10 trade gaps))) with a
//      placement-delta clause; (d) >= T executions more aggressive than our
//      price, against our side, counted from the join signal for an order
//      placed while it lives, optionally gated on queue position.
//      The NBBO is one call to the host's getNBBO(symst_idx); sigma comes
//      from the NBBO mid, or from this venue's round-lot BBO mid with
//      -DMD_CANCEL_VOL_FROM_VENUE=1 (getVenueRoundLotBBO);
//      -DMD_SIZE_UP_KEEPS_PRIORITY=1 for venues where a size increase keeps
//      time priority. Our orders
//      are flagged tracked in their level's QBlockList (qblocklist.hpp),
//      which supplies queue position and size. An order the host learns
//      is ours only after it entered the book (from the exchange's ack) is
//      tracked from then, its counts taken from the book as it stands
//      (on_place_late). Per-symbol order quads with
//      AVX-512VL kernels for 3+ live orders, no libm/div/sqrt on the
//      decision path; monotone per-symbol clock, generation-checked
//      handles, deferred callbacks.
//
//  [5] JoinTactic: signals when one order of at least a round lot improves
//      a side's best price on this venue (a higher bid, a lower offer);
//      stores the signal and this venue's round-lot far touch in
//      SymbolState::tac and counts the executions at or through the signal
//      price from then on, keeping enough of them for the cancel tactic to
//      start (d) for an order at any price. A signal lives until its kill
//      flag is read; new levels meanwhile are not new signals. One bit test
//      for a symbol it does not watch.
//
// Types follow the user's definitions exactly; PrcLevelGeneric drops the {}
// default member initializers so the type stays trivially default
// constructible (zeroed storage == valid empty state; required for static /
// hugepage placement, per earlier findings).
//
// Single writer. Build: -std=c++23 -O3 -march=native (AVX2; scalar fallback).

#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <immintrin.h>          // x86-64 only; SSE2 baseline is always present

#include "qblocklist.hpp"       // QBlockPool / QBlockList: per-level order queues ([4])

namespace md {

using PriceType      = int64_t;
using SizeType       = uint32_t;
using NumOrdersType  = uint16_t;
using FeedTimeType   = uint64_t;
using UniqueUpdateID = uint64_t;
using OrderRef       = uint64_t;

struct PrcLevelGeneric {          // trivially default constructible: zeroed
    FeedTimeType   feed_time;     // storage is a valid empty level
    UniqueUpdateID uuid;
    SizeType       agg_sz;
    NumOrdersType  num_orders;
    void*          plist;
};
static_assert(sizeof(PrcLevelGeneric) == 32);

enum class Side : uint8_t { Buy = 0, Sell = 1 };

// ---------------------------------------------------------------------------
// [1] OrderTable — 128B buckets, 4 orders each, one-vector key scan
// ---------------------------------------------------------------------------

class OrderTable {
public:
    static constexpr unsigned kLanes = 4;

    struct alignas(64) Bucket {
        // line 1: keys (one __m256i) + prices
        OrderRef  key[kLanes];      // 32B; 0 = free/deleted
        PriceType prc[kLanes];      // 32B
        // line 2: rest of the Order payload + chain link
        SizeType  sz[kLanes];       // 16B
        uint32_t  ordq[kLanes];     // 16B  level handle (bit31 = overflow)
        uint16_t  symst[kLanes];    // 8B
        uint8_t   side[kLanes];     // 4B
        uint32_t  next;             // 4B   pool index + 1; 0 = end of chain
        uint8_t   _pad[8];
    };
    static_assert(sizeof(Bucket) == 128, "bucket must be exactly 2 cache lines");

    struct Handle {                 // result of find(): mutate through it
        Bucket*  b = nullptr;
        unsigned lane = 0;
        explicit operator bool() const noexcept { return b != nullptr; }
    };

    // peak_resident orders; directory sized for ~2 orders/bucket average.
    // overflow_buckets = black-swan pool (all preallocated up front).
    explicit OrderTable(std::size_t peak_resident, std::size_t overflow_buckets) {
        const std::size_t want = (peak_resident + 1) / 2;
        nbuckets_ = std::bit_ceil(want < 8 ? std::size_t{8} : want);
        mask_     = nbuckets_ - 1;
        shift_    = 64u - static_cast<unsigned>(std::countr_zero(nbuckets_));
        dir_.assign(nbuckets_, Bucket{});
        pool_.assign(overflow_buckets ? overflow_buckets : 1, Bucket{});
    }

    // Insert a new order (ITCH refs are day-unique: caller guarantees absence).
    // Returns false only if the black-swan pool itself is exhausted.
    bool insert(OrderRef ref, PriceType prc, SizeType sz, uint32_t ordq,
                uint16_t symst, Side sd) noexcept {
        assert(ref != 0);
        Bucket* b = &dir_[bucket_of(ref)];
        for (;;) {
            const unsigned free = scan(b, OrderRef{0});
            if (free < kLanes) { write(b, free, ref, prc, sz, ordq, symst, sd); ++size_; return true; }
            if (!b->next) break;
            b = &pool_[b->next - 1];
        }
        const uint32_t ni = pool_alloc();                 // black swan: chain one more
        if (ni == 0) [[unlikely]] { ++insert_failures_; return false; }
        b->next = ni;
        Bucket* nb = &pool_[ni - 1];
        write(nb, 0, ref, prc, sz, ordq, symst, sd);
        ++size_;
        return true;
    }

    [[nodiscard]] Handle find(OrderRef ref) noexcept {
        assert(ref != 0);
        Bucket* b = &dir_[bucket_of(ref)];
        for (;;) {
            const unsigned lane = scan(b, ref);
            if (lane < kLanes) return {b, lane};
            if (!b->next) return {};
            b = &pool_[b->next - 1];
        }
    }

    // Feed-handler stage: warm both cache lines of the ref's home bucket the
    // moment the reference is decoded off the wire. The address depends only
    // on the hash — no dependent loads — so this can issue arbitrarily early.
    void prefetch(OrderRef ref) const noexcept {
        const Bucket* b = &dir_[bucket_of(ref)];
        __builtin_prefetch(b, 1, 3);                                    // keys + prices
        __builtin_prefetch(reinterpret_cast<const char*>(b) + 64, 1, 3); // payload
    }

    // Erase by ref; zeros the key (reusable). Unlinks+reclaims an overflow
    // bucket that becomes all-zero. Copies the payload out for the caller.
    bool erase(OrderRef ref, PriceType* prc, SizeType* sz, uint32_t* ordq,
               uint16_t* symst, uint8_t* sd) noexcept {
        assert(ref != 0);
        Bucket* prev = nullptr;
        Bucket* b = &dir_[bucket_of(ref)];
        for (;;) {
            const unsigned lane = scan(b, ref);
            if (lane < kLanes) {
                if (prc)   *prc   = b->prc[lane];
                if (sz)    *sz    = b->sz[lane];
                if (ordq)  *ordq  = b->ordq[lane];
                if (symst) *symst = b->symst[lane];
                if (sd)    *sd    = b->side[lane];
                b->key[lane] = 0;
                --size_;
                if (prev && all_free(b)) {                // reclaim empty overflow
                    prev->next = b->next;
                    pool_free(static_cast<uint32_t>(b - pool_.data()) + 1);
                }
                return true;
            }
            if (!b->next) return false;
            prev = b;
            b = &pool_[b->next - 1];
        }
    }

    [[nodiscard]] std::size_t size()            const noexcept { return size_; }
    [[nodiscard]] std::size_t buckets()         const noexcept { return nbuckets_; }
    [[nodiscard]] std::size_t pool_in_use()     const noexcept { return pool_used_; }
    [[nodiscard]] std::size_t pool_capacity()   const noexcept { return pool_.size(); }
    [[nodiscard]] std::size_t insert_failures() const noexcept { return insert_failures_; }
    [[nodiscard]] std::size_t max_chain() const noexcept {
        std::size_t worst = 1;
        for (const Bucket& d : dir_) {
            std::size_t n = 1;
            for (uint32_t x = d.next; x; x = pool_[x - 1].next) ++n;
            worst = worst < n ? n : worst;
        }
        return worst;
    }

private:
    static constexpr uint64_t kFib = 0x9E3779B97F4A7C15ULL;
    [[nodiscard]] std::size_t bucket_of(OrderRef r) const noexcept {
        return (r * kFib) >> shift_;
    }

    // lane of first key == what, or kLanes if none. One vector op on AVX2.
    [[nodiscard]] static unsigned scan(const Bucket* b, OrderRef what) noexcept {
#if defined(__AVX2__)
        const __m256i k = _mm256_load_si256(reinterpret_cast<const __m256i*>(b->key));
        const __m256i w = _mm256_set1_epi64x(static_cast<long long>(what));
        const int m = _mm256_movemask_pd(
            _mm256_castsi256_pd(_mm256_cmpeq_epi64(k, w)));
        return m ? static_cast<unsigned>(std::countr_zero(static_cast<unsigned>(m)))
                 : kLanes;
#else
        for (unsigned i = 0; i < kLanes; ++i) if (b->key[i] == what) return i;
        return kLanes;
#endif
    }

    [[nodiscard]] static bool all_free(const Bucket* b) noexcept {
#if defined(__AVX2__)
        const __m256i k = _mm256_load_si256(reinterpret_cast<const __m256i*>(b->key));
        const int m = _mm256_movemask_pd(
            _mm256_castsi256_pd(_mm256_cmpeq_epi64(k, _mm256_setzero_si256())));
        return m == 0xF;
#else
        for (unsigned i = 0; i < kLanes; ++i) if (b->key[i]) return false;
        return true;
#endif
    }

    static void write(Bucket* b, unsigned lane, OrderRef ref, PriceType prc,
                      SizeType sz, uint32_t ordq, uint16_t symst, Side sd) noexcept {
        b->key[lane]   = ref;
        b->prc[lane]   = prc;
        b->sz[lane]    = sz;
        b->ordq[lane]  = ordq;
        b->symst[lane] = symst;
        b->side[lane]  = static_cast<uint8_t>(sd);
    }

    uint32_t pool_alloc() noexcept {                       // returns index+1, 0 = none
        uint32_t i;
        if (free_head_) { i = free_head_; free_head_ = pool_[i - 1].next; }
        else if (pool_hw_ < pool_.size()) { i = static_cast<uint32_t>(++pool_hw_); }
        else return 0;
        Bucket& nb = pool_[i - 1];
        std::memset(nb.key, 0, sizeof(nb.key));
        nb.next = 0;
        ++pool_used_;
        return i;
    }
    void pool_free(uint32_t i) noexcept {
        pool_[i - 1].next = free_head_;
        free_head_ = i;
        --pool_used_;
    }

    std::vector<Bucket> dir_, pool_;
    std::size_t nbuckets_ = 0, mask_ = 0, size_ = 0;
    std::size_t pool_hw_ = 0, pool_used_ = 0, insert_failures_ = 0;
    uint32_t    free_head_ = 0;
    unsigned    shift_ = 0;
};

// ---------------------------------------------------------------------------
// [3] Paged ladder book (one side)
// ---------------------------------------------------------------------------

struct PagePool {
    static constexpr uint32_t kSlots = 512;                // levels per page
    struct Page { PrcLevelGeneric s[kSlots]; uint32_t live; };

    explicit PagePool(std::size_t npages) { pages_.assign(npages ? npages : 1, Page{}); }

    uint32_t alloc() noexcept {                            // index+1, 0 = none
        uint32_t i;
        if (!free_.empty()) { i = free_.back(); free_.pop_back(); }
        else if (hw_ < pages_.size()) i = static_cast<uint32_t>(++hw_);
        else return 0;
        Page& p = pages_[i - 1];
        p.live = 0;   // levels need no zeroing: fully written on creation and
        ++used_;      // never read before their occupancy bit is set
        return i;
    }
    void free(uint32_t i) noexcept { free_.push_back(i); --used_; }
    Page&       at(uint32_t i)       noexcept { return pages_[i - 1]; }
    const Page& at(uint32_t i) const noexcept { return pages_[i - 1]; }
    std::size_t in_use() const noexcept { return used_; }

    std::vector<Page> pages_;
    std::vector<uint32_t> free_;
    std::size_t hw_ = 0, used_ = 0;
};

// Secondary extents ("tiles"): small fixed-geometry paged regions allocated
// when a price lands outside every existing extent. Grid-aligned to absolute
// multiples of (kTicks * tick), so tiles never overlap each other; the router
// prefers the primary extent, so extent CONTENTS are always disjoint even
// where a tile's price range overlaps the primary's — which is what makes
// walking extents in base order, bit-scanning inside each, a strict
// price-ordered traversal with no merge logic.
struct TilePool {
    static constexpr uint32_t kTicks = 4096;      // ticks per tile extent
    struct Tile {
        PriceType base;                           // absolute price of tick 0
        uint64_t  occ[kTicks / 64];               // 64 occupancy words
        uint64_t  sum;                            // 1 summary word covers them
        uint32_t  dir[kTicks / 512];              // 8 page-pool slots
        uint32_t  live;                           // populated levels
        uint32_t  next;                           // freelist link (idx+1)
    };
    explicit TilePool(std::size_t n) {
        assert(n <= 65533 && "extent id must fit 16 bits below kBadHandle");
        tiles_.assign(n ? n : 1, Tile{});
    }
    uint16_t alloc(PriceType base) noexcept {      // idx+1; 0 = exhausted
        uint32_t i;
        if (free_head_) { i = free_head_; free_head_ = tiles_[i - 1].next; }
        else if (hw_ < tiles_.size()) i = static_cast<uint32_t>(++hw_);
        else return 0;
        Tile& t = tiles_[i - 1];
        std::memset(&t, 0, sizeof(Tile));
        t.base = base;
        ++used_;
        return static_cast<uint16_t>(i);
    }
    void free(uint16_t i) noexcept {
        tiles_[i - 1].next = free_head_; free_head_ = i; --used_;
    }
    Tile&       at(uint16_t i)       noexcept { return tiles_[i - 1]; }
    const Tile& at(uint16_t i) const noexcept { return tiles_[i - 1]; }
    std::size_t in_use() const noexcept { return used_; }

    std::vector<Tile> tiles_;
    uint32_t free_head_ = 0;
    std::size_t hw_ = 0, used_ = 0;
};

// Pure paged ladder: a large primary extent (anchored window, as before) plus
// small tile extents wherever price actually goes. No overflow container.
// Handle: ordq = (extent << 16) | tick_index; extent 0 = primary, else
// tile-pool index + 1. Reduce decodes the handle — still no price math.
//
// Contract the pure design imposes: `tick` must be the finest grid the venue
// can print for this symbol. An off-grid price (prc % tick != 0) has no slot
// in any extent; it is counted (offgrid_rejects) and add() returns
// kBadHandle. This is a configuration error class, like pool exhaustion —
// alarm on it, fix the symbol's tick.
class LadderBook {
public:
    static constexpr uint32_t kBadHandle = 0xFFFFFFFFu;

    LadderBook() = default;

    // window: primary extent tick count; pow2, 4096..65536 (idx fits 16 bits).
    void configure(bool is_bid, PriceType tick, uint32_t window) {
        assert(std::has_single_bit(window) && window >= 4096 && window <= 65536);
        assert(tick > 0);
        is_bid_ = is_bid;
        tick_   = tick;
        window_ = window;
        dir_.assign(window / PagePool::kSlots, 0);
        occ_.assign(window / 64, 0);
        sum_.assign(window / 4096, 0);
        anchor_ = -1;
        primary_live_ = 0;
        best_ext_ = kNoExt;
        tiles_.clear();
        tiles_.reserve(16);
        offgrid_ = tile_fail_ = 0;
    }

    // Add flow at price; creates level/tile as needed. Returns the handle, or
    // kBadHandle on off-grid price / tile-pool exhaustion (both counted).
    uint32_t add(PagePool& pp, TilePool& tp, PriceType prc, SizeType sz,
                 FeedTimeType ft, UniqueUpdateID uu) noexcept {
        if (prc % tick_ != 0) [[unlikely]] { ++offgrid_; return kBadHandle; }
        if (anchor_ < 0) {
            const PriceType half = static_cast<PriceType>(window_ / 2) * tick_;
            const PriceType a = prc > half ? prc - half : 0;
            anchor_ = (a / tick_) * tick_;          // grid-phase 0, like tiles
        }
        uint16_t e; uint32_t idx;
        const PriceType rel = prc - anchor_;
        if (rel >= 0 && rel / tick_ < static_cast<PriceType>(window_)) {
            e = 0; idx = static_cast<uint32_t>(rel / tick_);
        } else {                                    // outside primary -> tile
            const PriceType span = static_cast<PriceType>(TilePool::kTicks) * tick_;
            const PriceType base = (prc / span) * span;
            uint16_t te = tile_find(tp, base);
            if (!te) {
                te = tile_create(tp, base);
                if (!te) [[unlikely]] { ++tile_fail_; return kBadHandle; }
            }
            e = te; idx = static_cast<uint32_t>((prc - base) / tick_);
        }
        Ext x = ext_of(tp, e);
        const uint32_t di = idx >> 9, slot = idx & (PagePool::kSlots - 1);
        if (x.dir[di] == 0) {
            const uint32_t pi = pp.alloc();
            assert(pi != 0 && "page pool exhausted — size npages higher");
            x.dir[di] = pi;
        }
        PrcLevelGeneric& L = pp.at(x.dir[di]).s[slot];
        if (!bits_test(x.occ, idx)) {               // new level
            bits_set(x.occ, x.sum, idx);
            ++pp.at(x.dir[di]).live;
            if (e == 0) ++primary_live_; else ++tp.at(e).live;
            L.agg_sz = sz; L.num_orders = 1; L.plist = nullptr;
            if (best_ext_ == kNoExt ||
                (is_bid_ ? prc > best_prc_ : prc < best_prc_)) {
                best_ext_ = e; best_idx_ = idx; best_prc_ = prc;
            }
        } else {
            L.agg_sz += sz;
            assert(L.num_orders < static_cast<NumOrdersType>(~NumOrdersType{0}));
            L.num_orders += 1;
        }
        L.feed_time = ft; L.uuid = uu;
        return (static_cast<uint32_t>(e) << 16) | idx;
    }

    // Reduce through the handle (E/C/X/D path — no price math anywhere).
    void reduce(PagePool& pp, TilePool& tp, uint32_t ordq, SizeType sz,
                bool order_gone, FeedTimeType ft, UniqueUpdateID uu) noexcept {
        const uint16_t e   = static_cast<uint16_t>(ordq >> 16);
        const uint32_t idx = ordq & 0xFFFFu;
        Ext x = ext_of(tp, e);
        const uint32_t di = idx >> 9, slot = idx & (PagePool::kSlots - 1);
        assert(x.dir[di] != 0 && bits_test(x.occ, idx));
        PrcLevelGeneric& L = pp.at(x.dir[di]).s[slot];
        assert(L.agg_sz >= sz);
        L.agg_sz -= sz;
        L.feed_time = ft; L.uuid = uu;
        if (order_gone) {
            assert(L.num_orders > 0);
            if (--L.num_orders == 0) {              // level death
                assert(L.agg_sz == 0);
                bits_clear(x.occ, x.sum, idx);
                PagePool::Page& pg = pp.at(x.dir[di]);
                if (--pg.live == 0) { pp.free(x.dir[di]); x.dir[di] = 0; }
                if (e == 0) --primary_live_; else --tp.at(e).live;
                if (e == best_ext_ && idx == best_idx_)
                    recover_best(tp);               // before tile removal:
                if (e != 0 && tp.at(e).live == 0)   //  empty tile scans empty,
                    tile_remove(tp, e);             //  recovery steps past it
            }
        }
    }

    // Same-price replace fast path: the level survives with num_orders net
    // unchanged (old order out, new order in); only the aggregate moves. No
    // price math, no bitmap churn, no page or best_ activity.
    void resize_level(PagePool& pp, TilePool& tp, uint32_t ordq,
                      SizeType old_sz, SizeType new_sz,
                      FeedTimeType ft, UniqueUpdateID uu) noexcept {
        const uint16_t e   = static_cast<uint16_t>(ordq >> 16);
        const uint32_t idx = ordq & 0xFFFFu;
        Ext x = ext_of(tp, e);
        PrcLevelGeneric& L = pp.at(x.dir[idx >> 9]).s[idx & (PagePool::kSlots - 1)];
        assert(bits_test(x.occ, idx) && L.num_orders > 0 && L.agg_sz >= old_sz);
        L.agg_sz = L.agg_sz - old_sz + new_sz;
        L.feed_time = ft; L.uuid = uu;
    }

    // ---- queries ---------------------------------------------------------

    [[nodiscard]] std::optional<PriceType> best() const noexcept {
        if (best_ext_ == kNoExt) return std::nullopt;
        return best_prc_;                            // O(1): cache maintained
    }

    // Visit levels from the inside outward, strict price order across
    // extents. f(prc, const PrcLevelGeneric&) -> bool (false stops).
    template <class F>
    void for_each_from_best(const PagePool& pp, const TilePool& tp, F&& f) const {
        if (best_ext_ == kNoExt) return;
        auto* self = const_cast<LadderBook*>(this);          // single writer
        auto& tpm  = const_cast<TilePool&>(tp);
        uint16_t e = best_ext_; uint32_t idx = best_idx_; PriceType p = best_prc_;
        for (;;) {
            Ext x = self->ext_of(tpm, e);
            const PrcLevelGeneric& L =
                pp.at(x.dir[idx >> 9]).s[idx & (PagePool::kSlots - 1)];
            if (!f(p, L)) return;
            if (!self->advance_cursor(tpm, e, idx, p)) return;
        }
    }

    struct RlBest { PriceType price; uint64_t cum_size; uint32_t levels_used; };
    [[nodiscard]] std::optional<RlBest>
    roundlot_best(const PagePool& pp, const TilePool& tp, uint32_t lot) const {
        uint64_t cum = 0; uint32_t used = 0; std::optional<RlBest> out;
        for_each_from_best(pp, tp, [&](PriceType p, const PrcLevelGeneric& L) {
            cum += L.agg_sz; ++used;
            if (cum >= lot) { out = RlBest{p, cum, used}; return false; }
            return true;
        });
        return out;
    }

    // Level at an exact price, in whichever extent holds it.
    [[nodiscard]] const PrcLevelGeneric*
    level_at(const PagePool& pp, const TilePool& tp, PriceType prc) const noexcept {
        if (anchor_ < 0 || prc % tick_ != 0) return nullptr;
        const PriceType rel = prc - anchor_;
        if (rel >= 0 && rel / tick_ < static_cast<PriceType>(window_)) {
            const uint32_t idx = static_cast<uint32_t>(rel / tick_);
            if (!bits_test(occ_.data(), idx) || dir_[idx >> 9] == 0) return nullptr;
            return &pp.at(dir_[idx >> 9]).s[idx & (PagePool::kSlots - 1)];
        }
        const PriceType span = static_cast<PriceType>(TilePool::kTicks) * tick_;
        const PriceType base = (prc / span) * span;
        const uint16_t te = tile_find(tp, base);
        if (!te) return nullptr;
        const TilePool::Tile& t = tp.at(te);
        const uint32_t idx = static_cast<uint32_t>((prc - base) / tick_);
        if (!bits_test(t.occ, idx) || t.dir[idx >> 9] == 0) return nullptr;
        return &pp.at(t.dir[idx >> 9]).s[idx & (PagePool::kSlots - 1)];
    }

    // Prefetch resolvers (tick-processor stage): address without touching the
    // level line. nullptr where unresolvable — skip the prefetch.
    [[nodiscard]] const PrcLevelGeneric*
    level_addr(const PagePool& pp, const TilePool& tp, uint32_t ordq) const noexcept {
        const uint16_t e   = static_cast<uint16_t>(ordq >> 16);
        const uint32_t idx = ordq & 0xFFFFu;
        const uint32_t* dir = (e == 0) ? dir_.data() : tp.at(e).dir;
        if (dir[idx >> 9] == 0) return nullptr;
        return &pp.at(dir[idx >> 9]).s[idx & (PagePool::kSlots - 1)];
    }
    [[nodiscard]] const PrcLevelGeneric*
    level_addr_for_price(const PagePool& pp, const TilePool& tp,
                         PriceType prc) const noexcept {
        if (anchor_ < 0 || prc % tick_ != 0) return nullptr;
        const PriceType rel = prc - anchor_;
        if (rel >= 0 && rel / tick_ < static_cast<PriceType>(window_)) {
            const uint32_t idx = static_cast<uint32_t>(rel / tick_);
            if (dir_[idx >> 9] == 0) return nullptr;
            return &pp.at(dir_[idx >> 9]).s[idx & (PagePool::kSlots - 1)];
        }
        const PriceType span = static_cast<PriceType>(TilePool::kTicks) * tick_;
        const uint16_t te = tile_find(tp, (prc / span) * span);
        if (!te) return nullptr;
        const TilePool::Tile& t = tp.at(te);
        const uint32_t idx = static_cast<uint32_t>((prc - t.base) / tick_);
        if (t.dir[idx >> 9] == 0) return nullptr;
        return &pp.at(t.dir[idx >> 9]).s[idx & (PagePool::kSlots - 1)];
    }

    [[nodiscard]] std::size_t tiles_live()          const noexcept { return tiles_.size(); }
    [[nodiscard]] std::uint64_t offgrid_rejects()   const noexcept { return offgrid_; }
    [[nodiscard]] std::uint64_t tile_alloc_failures() const noexcept { return tile_fail_; }

private:
    static constexpr uint16_t kNoExt = 0xFFFF;

    // ---- generic two-level bitmap helpers --------------------------------
    static bool bits_test(const uint64_t* occ, uint32_t i) noexcept {
        return (occ[i >> 6] >> (i & 63)) & 1u;
    }
    static void bits_set(uint64_t* occ, uint64_t* sum, uint32_t i) noexcept {
        occ[i >> 6] |= 1ull << (i & 63);
        sum[i >> 12] |= 1ull << ((i >> 6) & 63);
    }
    static void bits_clear(uint64_t* occ, uint64_t* sum, uint32_t i) noexcept {
        occ[i >> 6] &= ~(1ull << (i & 63));
        if (occ[i >> 6] == 0) sum[i >> 12] &= ~(1ull << ((i >> 6) & 63));
    }
    static int32_t bits_next_below(const uint64_t* occ, const uint64_t* sum,
                                   uint32_t from) noexcept {
        int32_t w = static_cast<int32_t>(from >> 6);
        const uint32_t b = from & 63;
        uint64_t m = b ? (occ[w] & ((1ull << b) - 1)) : 0;
        if (m) return (w << 6) + 63 - std::countl_zero(m);
        int32_t sw = w >> 6;
        uint64_t sm = (w & 63) ? (sum[sw] & ((1ull << (w & 63)) - 1)) : 0;
        for (;;) {
            if (sm) {
                const int32_t w2 = (sw << 6) + 63 - std::countl_zero(sm);
                return (w2 << 6) + 63 - std::countl_zero(occ[w2]);
            }
            if (--sw < 0) return -1;
            sm = sum[sw];
        }
    }
    static int32_t bits_next_above(const uint64_t* occ, const uint64_t* sum,
                                   uint32_t nwords, uint32_t nsum,
                                   uint32_t from) noexcept {
        int32_t w = static_cast<int32_t>(from >> 6);
        const uint32_t b = from & 63;
        uint64_t m = (b == 63) ? 0 : (occ[w] & ~((2ull << b) - 1));
        if (m) return (w << 6) + std::countr_zero(m);
        int32_t sw = w >> 6;
        uint64_t sm = ((w & 63) == 63) ? 0 : (sum[sw] & ~((2ull << (w & 63)) - 1));
        for (;;) {
            if (sm) {
                const int32_t w2 = (sw << 6) + std::countr_zero(sm);
                if (w2 < static_cast<int32_t>(nwords))
                    return (w2 << 6) + std::countr_zero(occ[w2]);
            }
            if (++sw >= static_cast<int32_t>(nsum)) return -1;
            sm = sum[sw];
        }
    }
    static int32_t bits_highest(const uint64_t* occ, const uint64_t* sum,
                                uint32_t nsum) noexcept {
        for (int32_t sw = static_cast<int32_t>(nsum) - 1; sw >= 0; --sw)
            if (sum[sw]) {
                const int32_t w = (sw << 6) + 63 - std::countl_zero(sum[sw]);
                return (w << 6) + 63 - std::countl_zero(occ[w]);
            }
        return -1;
    }
    static int32_t bits_lowest(const uint64_t* occ, const uint64_t* sum,
                               uint32_t nsum) noexcept {
        for (int32_t sw = 0; sw < static_cast<int32_t>(nsum); ++sw)
            if (sum[sw]) {
                const int32_t w = (sw << 6) + std::countr_zero(sum[sw]);
                return (w << 6) + std::countr_zero(occ[w]);
            }
        return -1;
    }

    // ---- extent plumbing --------------------------------------------------
    struct Ext {
        uint32_t* dir; uint64_t* occ; uint64_t* sum;
        uint32_t nwords, nsum; PriceType base;
    };
    Ext ext_of(TilePool& tp, uint16_t e) noexcept {
        if (e == 0)
            return {dir_.data(), occ_.data(), sum_.data(),
                    static_cast<uint32_t>(occ_.size()),
                    static_cast<uint32_t>(sum_.size()), anchor_};
        TilePool::Tile& t = tp.at(e);
        return {t.dir, t.occ, &t.sum, TilePool::kTicks / 64, 1, t.base};
    }
    [[nodiscard]] uint16_t tile_find(const TilePool& tp, PriceType base) const noexcept {
        for (uint16_t te : tiles_) if (tp.at(te).base == base) return te;
        return 0;
    }
    uint16_t tile_create(TilePool& tp, PriceType base) noexcept {
        const uint16_t te = tp.alloc(base);
        if (!te) return 0;
        auto it = tiles_.begin();                     // keep sorted by base
        while (it != tiles_.end() && tp.at(*it).base < base) ++it;
        tiles_.insert(it, te);
        return te;
    }
    void tile_remove(TilePool& tp, uint16_t e) noexcept {
        for (auto it = tiles_.begin(); it != tiles_.end(); ++it)
            if (*it == e) { tiles_.erase(it); break; }
        tp.free(e);
    }

    // Advance a (extent, idx) cursor one populated level outward in price
    // order: scan within the extent, else step to the next non-empty extent
    // by base and take its extreme. Alloc-free; O(tiles) worst on the step.
    bool advance_cursor(TilePool& tp, uint16_t& e, uint32_t& idx,
                        PriceType& prc) noexcept {
        Ext x = ext_of(tp, e);
        const int32_t r = is_bid_
            ? bits_next_below(x.occ, x.sum, idx)
            : bits_next_above(x.occ, x.sum, x.nwords, x.nsum, idx);
        if (r >= 0) {
            idx = static_cast<uint32_t>(r);
            prc = x.base + static_cast<PriceType>(idx) * tick_;
            return true;
        }
        PriceType cur_base = x.base;
        for (;;) {
            uint16_t ne = 0; PriceType nb = 0; bool found = false;
            auto consider = [&](uint16_t ce, PriceType cb, bool nonempty) {
                if (!nonempty) return;
                if (is_bid_ ? (cb < cur_base && (!found || cb > nb))
                            : (cb > cur_base && (!found || cb < nb))) {
                    ne = ce; nb = cb; found = true;
                }
            };
            consider(0, anchor_, primary_live_ > 0 && anchor_ >= 0);
            for (uint16_t te : tiles_)
                consider(te, tp.at(te).base, tp.at(te).live > 0);
            if (!found) return false;
            Ext nx = ext_of(tp, ne);
            const int32_t ridx = is_bid_
                ? bits_highest(nx.occ, nx.sum, nx.nsum)
                : bits_lowest(nx.occ, nx.sum, nx.nsum);
            if (ridx >= 0) {
                e = ne; idx = static_cast<uint32_t>(ridx);
                prc = nx.base + static_cast<PriceType>(idx) * tick_;
                return true;
            }
            cur_base = nb;                            // defensive; unreachable
        }
    }
    void recover_best(TilePool& tp) noexcept {
        uint16_t e = best_ext_; uint32_t idx = best_idx_; PriceType p = 0;
        if (advance_cursor(tp, e, idx, p)) {
            best_ext_ = e; best_idx_ = idx; best_prc_ = p;
        } else {
            best_ext_ = kNoExt;
        }
    }

    bool       is_bid_ = true;
    PriceType  tick_   = 100;
    PriceType  anchor_ = -1;
    uint32_t   window_ = 32768;
    std::vector<uint32_t> dir_;
    std::vector<uint64_t> occ_, sum_;
    uint32_t   primary_live_ = 0;
    std::vector<uint16_t> tiles_;                     // sorted by base
    uint16_t   best_ext_ = kNoExt;
    uint32_t   best_idx_ = 0;
    PriceType  best_prc_ = 0;
    std::uint64_t offgrid_ = 0, tile_fail_ = 0;
};

// ---------------------------------------------------------------------------
// [2] SymbolState + system facade
// ---------------------------------------------------------------------------

// State the join tactic ([5]) writes and the cancel tactic ([4]) reads, kept
// with the symbol. Per side, one line: the side's join signal — set when one
// order of at least a round lot improved that side's best on this venue — and
// the executions at or through the signal price since; and two more lines,
// JoinTape, with the 16 most aggressive execution prices since the signal by
// resting side, from which the cancel tactic starts (d) for an order of ours
// at any price. A signal lives until the join tactic reads its kill flag.
struct alignas(64) JoinSide {
    PriceType    px = 0;           // signal price: the improving order's; 0 = no live signal
    PriceType    far_px = 0;       // this venue's round-lot far touch at the signal; 0 = none
    FeedTimeType t = 0;            // signal time: the improving order's Add
    uint32_t     sz = 0;           // the improving order's size (>= round lot)
    uint32_t     far_sz = 0;       // shares from the far side's top through far_px (>= round lot)
    uint32_t     lot = 100;        // round lot, copied from SymbolState::round_lot by arm()
    uint16_t     seq = 0;          // signals on this side so far (wraps)
    uint8_t      kill = 0;         // set by the strategy (for now JoinTactic::kill()): the join
                                   //   tactic ends the signal when it reads it; relaxed atomics
    uint8_t      pad_ = 0;
    uint32_t     n_at = 0;         // executions at px since the signal
    uint32_t     n_thru_same = 0;  //   beyond px (more aggressive), against this side's orders
    uint32_t     n_thru_opp = 0;   //   beyond px, against the other side's orders
    uint32_t     sh_at = 0, sh_thru_same = 0, sh_thru_opp = 0;   // their shares
};
static_assert(sizeof(JoinSide) == 64, "one line per side");

// Execution prices since a side's signal, as offsets from the signal price
// oriented so that larger is more aggressive (a buy signal: px - signal; a
// sell signal: signal - px), clamped to int32. Each array keeps the 16
// largest, descending, kTapeEmpty below them; `same` holds executions against
// resting orders on the signal's side, `opp` against the other side's. The
// number of executions more aggressive than a price is then exact up to 16
// per array — all (d) needs while aggr_prints_n <= 16.
inline constexpr int32_t kTapeEmpty = std::numeric_limits<int32_t>::min();
struct alignas(64) JoinTape {
    int32_t same[16];
    int32_t opp[16];
};
static_assert(sizeof(JoinTape) == 128, "two lines per side");
struct TacticShared {
    JoinSide side[2];              // [Buy], [Sell]
    JoinTape tape[2];
    TacticShared() noexcept { for (JoinTape& t : tape) clear_tape(t); }
    static void clear_tape(JoinTape& t) noexcept {
        for (int i = 0; i < 16; ++i) t.same[i] = t.opp[i] = kTapeEmpty;
    }
};

// Offset of px from signal price s0 on the tape (see JoinTape); never
// kTapeEmpty.
inline int32_t tape_off(PriceType px, PriceType s0, bool sell) noexcept {
    const int64_t d = sell ? static_cast<int64_t>(s0) - static_cast<int64_t>(px)
                           : static_cast<int64_t>(px) - static_cast<int64_t>(s0);
    return static_cast<int32_t>(std::clamp<int64_t>(d, int64_t{kTapeEmpty} + 1,
                                                    std::numeric_limits<int32_t>::max()));
}
// Entries of a tape array above offset o (more aggressive than it). Both
// tape kernels work on two ymm halves, as the cancel tactic's do: no 512-bit
// instruction on the execution path, so no license or upper-lane power-up
// transition when a quiet symbol trades. Unaligned loads and stores (free on
// aligned data): the host's SymbolState storage need not honour alignas.
inline uint32_t tape_above(const int32_t* v, int32_t o) noexcept {
#if defined(__AVX512VL__)
    const __m256i* p = reinterpret_cast<const __m256i*>(v);
    const __m256i ov = _mm256_set1_epi32(o);
    return static_cast<uint32_t>(std::popcount(
        static_cast<unsigned>(_mm256_cmpgt_epi32_mask(_mm256_loadu_si256(p), ov)) |
        (static_cast<unsigned>(_mm256_cmpgt_epi32_mask(_mm256_loadu_si256(p + 1), ov)) << 8)));
#else
    uint32_t n = 0;
    for (int i = 0; i < 16; ++i) n += v[i] > o ? 1u : 0u;
    return n;
#endif
}
// Keep x if it is among the 16 largest (ties go after their equals). The
// compare says where it goes (the entries below x are a suffix: v
// descends); vpexpandd shifts them down one around it, and valignd carries
// the low half's last entry into the high half.
inline void tape_insert(int32_t* v, int32_t x) noexcept {
#if defined(__AVX512VL__)
    __m256i* p = reinterpret_cast<__m256i*>(v);
    const __m256i lo = _mm256_loadu_si256(p), hi = _mm256_loadu_si256(p + 1), xv = _mm256_set1_epi32(x);
    const unsigned m = static_cast<unsigned>(_mm256_cmpgt_epi32_mask(xv, lo)) |
                       (static_cast<unsigned>(_mm256_cmpgt_epi32_mask(xv, hi)) << 8);
    if (!m) return;
    const unsigned i = static_cast<unsigned>(std::countr_zero(m));
    if (i < 8) {
        _mm256_storeu_si256(p + 1, _mm256_alignr_epi32(hi, lo, 7));
        _mm256_storeu_si256(p, _mm256_mask_expand_epi32(xv, static_cast<__mmask8>(~(1u << i)), lo));
    } else {
        _mm256_storeu_si256(p + 1, _mm256_mask_expand_epi32(xv, static_cast<__mmask8>(~(1u << (i - 8))), hi));
    }
#else
    if (!(x > v[15])) return;
    int i = 15;
    for (; i > 0 && x > v[i - 1]; --i) v[i] = v[i - 1];
    v[i] = x;
#endif
}
// The side's live, unkilled signal, if any.
inline const JoinSide* join_live(const TacticShared& t, unsigned side) noexcept {
    const JoinSide& js = t.side[side];
    return js.px != 0 && !__atomic_load_n(&js.kill, __ATOMIC_RELAXED) ? &js : nullptr;
}
// Executions more aggressive than px since the side's live signal: against
// the side's resting orders, plus the other side's unless same_only; 0 if
// the side has no live signal or its kill flag is set. Exact up to 16 each.
// Inlined into its two callers, on_place and aggr_since, which are calls.
[[gnu::always_inline]] inline uint32_t join_aggr(const TacticShared& t, unsigned side, PriceType px,
                                                 bool same_only) noexcept {
    const JoinSide* js = join_live(t, side);
    if (!js) return 0;
    const int32_t o = tape_off(px, js->px, side != 0);
    const JoinTape& tp = t.tape[side];
    return tape_above(tp.same, o) + (same_only ? 0u : tape_above(tp.opp, o));
}

struct SymbolState {
    LadderBook   books[2];         // [Buy], [Sell]
    PriceType    tick        = 100;
    uint32_t     round_lot   = 100;
    uint8_t      market_state = 0; // trading/halted/... user-defined
    TacticShared tac;              // join and cancel tactics ([4], [5])
};

class MarketDataSystem {
public:
    static constexpr std::size_t kMaxSymbols = 1u << 14;   // symst_idx < 2^14

    MarketDataSystem(std::size_t nsymbols, std::size_t peak_orders,
                     std::size_t npages, std::size_t swan_buckets,
                     std::size_t ntiles = 4096)
        : orders_(peak_orders, swan_buckets), pages_(npages), tiles_(ntiles),
          syms_(nsymbols) {
        assert(nsymbols <= kMaxSymbols);
        for (auto& s : syms_) {
            s.books[0].configure(true,  s.tick, 32768);
            s.books[1].configure(false, s.tick, 32768);
        }
    }

    void configure_symbol(uint16_t symst_idx, PriceType tick, uint32_t round_lot,
                          uint32_t window = 32768) {
        SymbolState& s = sym(symst_idx);
        s.tick = tick; s.round_lot = round_lot;
        s.books[0].configure(true,  tick, window);
        s.books[1].configure(false, tick, window);
    }

    // ---- ITCH-shaped message handlers -----------------------------------

    bool on_add(OrderRef ref, uint16_t symst_idx, Side sd, PriceType prc,
                SizeType sz, FeedTimeType ft, UniqueUpdateID uu) noexcept {
        SymbolState& s = sym(symst_idx);
        const uint32_t ordq = s.books[static_cast<unsigned>(sd)]
                                  .add(pages_, tiles_, prc, sz, ft, uu);
        if (ordq == LadderBook::kBadHandle) [[unlikely]] return false;
        return orders_.insert(ref, prc, sz, ordq, symst_idx, sd);
    }

    // Execute/cancel `sz` shares; removes the order if it reaches zero.
    bool on_reduce(OrderRef ref, SizeType sz,
                   FeedTimeType ft, UniqueUpdateID uu) noexcept {
        auto h = orders_.find(ref);
        if (!h) return false;
        return on_reduce_h(h, ref, sz, ft, uu);
    }

    // Reduce through a pre-verified handle (batched pipeline fast path).
    // Caller must have checked h.b->key[h.lane] == ref after any intervening
    // mutations.
    bool on_reduce_h(OrderTable::Handle h, OrderRef ref, SizeType sz,
                     FeedTimeType ft, UniqueUpdateID uu) noexcept {
        assert(h && h.b->key[h.lane] == ref);
        assert(h.b->sz[h.lane] >= sz);
        const SizeType remain = h.b->sz[h.lane] - sz;
        SymbolState& s = sym(h.b->symst[h.lane]);
        LadderBook& bk = s.books[h.b->side[h.lane]];
        if (remain == 0) {
            bk.reduce(pages_, tiles_, h.b->ordq[h.lane], sz, /*order_gone*/true, ft, uu);
            return orders_.erase(ref, nullptr, nullptr, nullptr, nullptr, nullptr);
        }
        bk.reduce(pages_, tiles_, h.b->ordq[h.lane], sz, /*order_gone*/false, ft, uu);
        h.b->sz[h.lane] = remain;
        return true;
    }

    bool on_delete(OrderRef ref, FeedTimeType ft, UniqueUpdateID uu) noexcept {
        PriceType prc; SizeType sz; uint32_t ordq; uint16_t sy; uint8_t sd;
        if (!orders_.erase(ref, &prc, &sz, &ordq, &sy, &sd)) return false;
        sym(sy).books[sd].reduce(pages_, tiles_, ordq, sz, /*order_gone*/true, ft, uu);
        return true;
    }

    // ITCH U: delete old ref, add new ref at new prc/sz on the same side.
    bool on_replace(OrderRef old_ref, OrderRef new_ref, PriceType prc,
                    SizeType sz, FeedTimeType ft, UniqueUpdateID uu) noexcept {
        PriceType op; SizeType osz; uint32_t oq; uint16_t sy; uint8_t sd;
        if (!orders_.erase(old_ref, &op, &osz, &oq, &sy, &sd)) return false;
        LadderBook& bk = sym(sy).books[sd];
        if (prc == op && sz > 0) {          // price unchanged: delta the level
            bk.resize_level(pages_, tiles_, oq, osz, sz, ft, uu);
            return orders_.insert(new_ref, prc, sz, oq, sy, static_cast<Side>(sd));
        }
        bk.reduce(pages_, tiles_, oq, osz, /*order_gone*/true, ft, uu);
        const uint32_t nq = bk.add(pages_, tiles_, prc, sz, ft, uu);
        if (nq == LadderBook::kBadHandle) [[unlikely]] return false;
        return orders_.insert(new_ref, prc, sz, nq, sy, static_cast<Side>(sd));
    }

    // ---- batched pipeline: feed-handler + tick-processor prefetch --------

    struct TickMsg {
        enum class T : uint8_t { Add, Reduce, Delete, Replace } type;
        OrderRef  ref;         // order reference (Replace: the OLD ref)
        OrderRef  ref2;        // Replace: the new reference
        uint16_t  symst;       // Add only
        Side      side;        // Add only
        PriceType prc;         // Add / Replace(new) price
        SizeType  sz;          // Add/Replace shares; Reduce: shares removed
        FeedTimeType   ft;
        UniqueUpdateID uu;
    };

    // Process one packet's messages in order, with two prefetch stages:
    //   stage 1 (feed handler): warm every message's order bucket as soon as
    //     refs are known — pure hash arithmetic, issued for the whole packet
    //     before any processing, giving maximal distance.
    //   stage 2 (tick processor): one message of lookahead — resolve msg
    //     i+1's order handle and prefetch its price level while msg i's book
    //     mutation executes.
    // Hazard: msg i can invalidate the handle prepared for i+1 (same-ref
    // sequences in one packet, or bucket reclaim+reuse). Handles are
    // re-verified via key[lane]==ref before use; mismatch falls back to a
    // fresh find on the now-hot bucket. Refs are day-unique and pool buckets
    // are never deallocated, so a stale handle can never verify against the
    // wrong order. Results are identical to the per-message handlers.
    void process_batch(const TickMsg* m, std::size_t n) noexcept {
        for (std::size_t i = 0; i < n; ++i) {              // stage 1
            orders_.prefetch(m[i].ref);
            if (m[i].type == TickMsg::T::Replace) orders_.prefetch(m[i].ref2);
        }
        auto prepare = [&](const TickMsg& x) -> OrderTable::Handle {
            if (x.type == TickMsg::T::Add) {
                const auto* L = sym(x.symst).books[static_cast<unsigned>(x.side)]
                                    .level_addr_for_price(pages_, tiles_, x.prc);
                if (L) __builtin_prefetch(L, 1, 3);
                return {};
            }
            auto h = orders_.find(x.ref);
            if (h) {
                const SymbolState& s = sym(h.b->symst[h.lane]);
                const LadderBook& bk = s.books[h.b->side[h.lane]];
                if (const auto* L = bk.level_addr(pages_, tiles_, h.b->ordq[h.lane]))
                    __builtin_prefetch(L, 1, 3);
                if (x.type == TickMsg::T::Replace)          // new level too
                    if (const auto* L2 = bk.level_addr_for_price(pages_, tiles_, x.prc))
                        __builtin_prefetch(L2, 1, 3);
            }
            return h;
        };
        OrderTable::Handle la{};
        if (n) la = prepare(m[0]);
        for (std::size_t i = 0; i < n; ++i) {              // stage 2
            const OrderTable::Handle h = la;
            if (i + 1 < n) la = prepare(m[i + 1]);         // overlaps apply(i)
            const TickMsg& x = m[i];
            switch (x.type) {
            case TickMsg::T::Add:
                on_add(x.ref, x.symst, x.side, x.prc, x.sz, x.ft, x.uu);
                break;
            case TickMsg::T::Reduce:
                if (h && h.b->key[h.lane] == x.ref)        // verify, then fast path
                    on_reduce_h(h, x.ref, x.sz, x.ft, x.uu);
                else
                    on_reduce(x.ref, x.sz, x.ft, x.uu);
                break;
            case TickMsg::T::Delete:
                on_delete(x.ref, x.ft, x.uu);
                break;
            case TickMsg::T::Replace:
                on_replace(x.ref, x.ref2, x.prc, x.sz, x.ft, x.uu);
                break;
            }
        }
    }

    // ---- queries ---------------------------------------------------------

    [[nodiscard]] std::optional<PriceType> best(uint16_t symst_idx, Side sd) const noexcept {
        return csym(symst_idx).books[static_cast<unsigned>(sd)].best();
    }
    [[nodiscard]] std::optional<LadderBook::RlBest>
    roundlot_best(uint16_t symst_idx, Side sd) const {
        const SymbolState& s = csym(symst_idx);
        return s.books[static_cast<unsigned>(sd)].roundlot_best(pages_, tiles_, s.round_lot);
    }
    template <class F>
    void for_each_level(uint16_t symst_idx, Side sd, F&& f) const {
        csym(symst_idx).books[static_cast<unsigned>(sd)]
            .for_each_from_best(pages_, tiles_, static_cast<F&&>(f));
    }
    [[nodiscard]] const PrcLevelGeneric*
    level_at(uint16_t symst_idx, Side sd, PriceType prc) const noexcept {
        return csym(symst_idx).books[static_cast<unsigned>(sd)].level_at(pages_, tiles_, prc);
    }

    OrderTable&       order_table()       noexcept { return orders_; }
    const OrderTable& order_table() const noexcept { return orders_; }
    const PagePool&   page_pool()   const noexcept { return pages_; }
    const TilePool&   tile_pool()   const noexcept { return tiles_; }
    SymbolState&      sym(uint16_t i)        noexcept { assert(i < syms_.size()); return syms_[i]; }
    const SymbolState& csym(uint16_t i) const noexcept { assert(i < syms_.size()); return syms_[i]; }

private:
    OrderTable  orders_;
    PagePool    pages_;
    TilePool    tiles_;
    std::vector<SymbolState> syms_;
};

// ---------------------------------------------------------------------------
// [4] CancelTactic — resting-order cancel decisions
//
// Every check is triggered by a book update the venue feed delivers for the
// order's symbol (add, delete, modify, cancel, replace, execute), and only
// for a TRACKED symbol: one with active (undecided) orders, or one the host
// pinned with set_tracking(). For any other symbol on_book_update returns
// after one bit test — no NBBO read, no estimator update, no check. For a
// tracked symbol it reads the NBBO with the host's getNBBO(sym), re-checks
// that symbol's resting orders (ours), on an execution counts (d), and when
// an order leaves one of our levels, or a market order there is cut, checks
// (b). Four independent triggers:
//
//   (a) QUEUE POSITION.  Cancel when at least queue_n orders and at least
//       queue_k x our size in shares are ahead of ours in its level's queue
//       (defaults 5 and 3). Nothing can join ahead of a resting order, so
//       on_place checks it and reports it at once, and it is re-checked only
//       when the shares ahead can have grown against our size: when one of
//       our orders at that price is reduced, and, on venues where a size
//       increase keeps time priority (MD_SIZE_UP_KEEPS_PRIORITY), when an
//       order there — a market order or one of ours — grows in place.
//
//   (b) LEADING AN ABANDONED QUEUE.  Our leading order at a price — the
//       first of ours in that level's queue — is at the front or has only
//       odd lots ahead of it (no order ahead holds a round lot; what is or
//       was behind it does not matter), and more than abandon_pct %
//       (default 50) of the market orders that were ahead of it at placement
//       holding at least a round lot left by cancellation (delete, cancel to
//       zero, replace) rather than by execution. Then every live order of
//       ours at that price is decided with it (CancelAbandoned for each),
//       whatever is ahead of the others. Only the leading order is judged:
//       an order of ours behind it becomes the leading one when it leaves
//       the book, and is judged then, on its own counts. on_place counts
//       an order's round lots ahead and flags them in the queue (kQAhead),
//       and keeps its count of orders ahead; on each removal at its price
//       that count says whether the order that left was ahead of it, and
//       the flags and event say how it left. Our own orders ahead never
//       count as cancellations. The flags are one bit per market order: a
//       new order's scan rewrites them back to the nearest order of ours
//       ahead of it (a round lot now, or not), and from that order on only
//       adds — so an order of ours behind another takes over the flags of
//       the orders ahead of that one as they stand. Checked when an order
//       leaves the price and when a flagged market order there is cut (the
//       last round lot ahead of ours worked down to an odd lot), for an
//       active leading order: a decided one (its cancel in flight) neither
//       decides nor counts until rearm(), and while it rests no other order
//       of ours there is judged.
//
//   (c) TOO DEEP.  depth = |far_touch - px|   (far touch: NBB offer for a
//       buy, NBB bid for a sell — the distance to the price we could trade
//       at right now). Cancel when
//           depth > thr,  thr = max( hist_spread_mult * hist_spread,
//                                    spread_now,
//                                    min_ticks * tick,
//                                    sigma * sqrt(min(cap, gap_mult*tau)) )
//       and the market has moved away since placement:
//           depth - depth_at_place > place_spread_mult * spread_at_place.
//       hist_spread is fixed per symbol, set at startup (set_symbol), so the
//       first and third floors are one precomputed constant. An order at the
//       inside has depth == spread, so the spread floors make it un-deep by
//       construction. The sigma term is the one-sided fill-probability proxy
//       over a horizon of ~10 trade events, capped. Stability gate: the far
//       touch must have been unchanged for far_touch_min_stable_ns before it
//       is trusted, so the update that first sees the far touch move never
//       fires (c) itself; the decision lands on the first book update for the
//       symbol after the window.
//
//   (d) OUT-COMPETED.  Count executions after placement at a price more
//       aggressive than ours (above a buy, below a sell). An execution names
//       the resting side it traded against, so the aggressor is exact: an
//       execution against a resting bid was seller-initiated. With
//       passive_side_only, count only executions against our side — bids
//       priced above our buy being hit while we sit, offers below our sell
//       being lifted. An order placed while its side's join signal lives
//       (SymbolState::tac, [5]) starts from the executions more aggressive
//       than its own price since the signal, at any price. Cancel at >=
//       aggr_prints_n (T), optionally gated on a poor queue position: at
//       least aggr_gate_n orders and aggr_gate_sz shares ahead of ours.
//
// Orders live in the book. Our resting orders are venue orders like any
// other, held in time priority in their price level's QBlockList
// (qblocklist.hpp; list state in PrcLevelGeneric::plist). on_place takes the
// order's queue handle and flags its slot tracked; the book's remove() and
// reduce() report the flag, so the E/C/X/D path learns "one of ours" from
// the block header it already touched, and calls on_book_remove() when a
// tracked order leaves the book (filled, canceled, replaced). Queue position
// and remaining size are read from the queue when needed ((a), (b), the (d)
// gate, order()), never copied, so the tactic keeps no
// size and needs no fill callbacks. The host passes the flags its remove(),
// reduce() or grow() returned with each message, which is how (b) learns
// that a flagged order left or was cut, and how, and (a) that the queue
// changed against one of ours.
//
// Orders recognised late. on_place assumes the host knows an order is ours
// when its Add arrives. When the exchange's acknowledgement, which tells it,
// arrives after the Add, the host calls on_place_late() at the ack instead:
// the order is tracked from then, with the queue and the NBBO as they stand
// in place of what they were at the Add. Nothing is kept for it beforehand;
// see on_place_late for what that approximation gives up.
//
// NBBO. getNBBO(symst_idx) is the host's: an atomic snapshot of both sides
// of the best known NBBO (declared below, defined by the host system). The
// tactic reads it once per book update of a tracked symbol — and once in
// on_place and replace_allowed — and keeps the pair it last saw. A pair
// that differs is a quote change, stamped with the timestamp of the book
// update that observed it, so the stability window starts when a change is
// SEEN (never earlier than it happened) and, by default, the variance
// estimator samples the NBBO at this venue's book updates. The venue's own
// book is NOT the reference for (c): on an exchange that is not at the
// inside, the far touch would be wrong by the whole inter-venue spread. (The
// volatility alone can be taken from the venue's own round-lot BBO
// instead: see VolSource.)
//
// Clock. All timestamps are venue feed time. The symbol clock is the max
// book-update timestamp seen, so an early-stamped update is clamped to the
// previous instant and no interval runs backwards. on_place's ts marks
// which executions count toward (d): strictly later ones (compared on the
// execution's own timestamp). Every subtraction is guarded: nothing wraps.
//
// Estimators (time-decayed; they run only while the symbol is tracked):
//   sigma  variance RATE: exp-weighted sum of squared mid changes over
//          exp-weighted elapsed ns (price^2/ns, so sigma*sqrt(T) needs no
//          unit conversion). Mid changes sharing one timestamp collapse to
//          their net move. An untracked stretch enters as one sample: the mid
//          change across it, over its length. The mid is the NBBO's by
//          default; with VolSource::Venue (-DMD_CANCEL_VOL_FROM_VENUE=1) it
//          is that of this venue's own round-lot BBO, read through
//          getVenueRoundLotBBO on every book update (after its decision
//          pass), so every change of it is seen at its own timestamp.
//   tau    EWMA of gaps between trade EVENTS: executions within
//          trade_coalesce_ns of the event's first (default 0: the same
//          timestamp — one aggressive order filling several resting orders)
//          are one event. Seeded with the first gap. Executions are not seen
//          while untracked, so when tracking resumes the next execution
//          starts a new gap instead of closing one across the stretch.
//
// Lifecycle and callbacks. on_place returns a Handle (generation << 32 |
// slot) and every lifecycle call validates it: a stale handle — e.g. a
// cancel ack arriving after the order already left the book and its lane
// was reused — is a no-op returning false. Decisions are recorded as
// per-symbol pending bits during the pass and delivered after it, so the
// callback may call on_done / on_place / rearm / on_book_remove or feed
// more book updates; a decision whose order an earlier callback freed is
// dropped (freeing clears its pending bit). The tactic only DECIDES: the
// order is marked inactive so it cannot fire twice; the lane is released
// by on_book_remove() (the order left the book) or on_done(), and rearm()
// re-activates it after a cancel reject.
//
// Latency design (Granite Rapids, GCC 14, -march=graniterapids). The figures
// quoted in these comments were measured on a Granite Rapids VM with g++ 13.3
// (mdsys_2_cancel_bench.cpp, mdsys_2_host_bench.cpp); what a compiler inlines
// where is its own choice, which is why the attributes below leave it none
// on the paths that matter.
//   Gate.  The tracking bits are one 2 KB bitmap for 16384 symbols, so the
//     test that skips an untracked symbol touches no per-symbol state.
//   Layout.  Per symbol, a 3-line state block — line 0 everything a decision
//     reads first, line 1 sigma and tau, line 2 hysteresis and configuration
//     — and per side two 4-order quads, field-major within the quad, and
//     one line of (b) state for the side's 8 lanes:
//         line 0   px[4] | move_min[4]                  read by (c), (b)'s match
//         line 1   t_place[4] | n_aggr[4] | qh[4]        read by (d)
//         line 2   id / generation                       read when firing
//         (b)      orders ahead, n0, nc for 8 lanes      read on a removal at an
//                                                        armed price
//     One array per field across all 8 lanes would put a lone order's fields
//     on 4-5 lines; quads keep a side with <= 4 orders to 1-3 lines, and each
//     64-bit field of a quad is exactly one ymm. No lists:
//     placement is a tzcnt over the free-lane mask, removal clears a bit. At most
//     kLanes = 8 tracked orders per symbol per side; on_place returns
//     kInvalid beyond that, or past max_orders in total.
//   Fetch order.  Every load a decision needs is either at an address known
//     from sym alone or prefetched at one (prefetch_quads), so a cold
//     decision is one memory round trip, not three (state -> lanes ->
//     id/generation). The getNBBO load depends on no tactic state, so it
//     overlaps that round trip. prefetch(sym) does the same from the decode
//     stage; with the host prefetching its own NBBO record there as well, a
//     cold update or decision costs about what a hot one does (measured, one
//     order a side: an update 158-168 -> 36-54 TSC, a (d) cancel to the
//     callback 199 -> 69, hot 65; with the tactic's prefetch alone the NBBO
//     record's miss stays on the path, and they cost no less: 172-192 and
//     189). The queue is read at placement ((a), (b)), by the (d) gate after
//     a count reaches aggr_prints_n, for (b)'s count of orders ahead on a
//     removal at an armed price — and, once an order's ratio is met, for the
//     sizes ahead of it — and for (a) when the queue there changed against
//     ours.
//   Removals.  With (b) on, most removals of a symbol we rest in find it
//     armed. The armed lanes' prices are compared in line: two xmm compares
//     per quad, at addresses known from sym and side, so the loads do not
//     wait for the Sym line — about 4 TSC. xmm rather than ymm because this
//     part is inlined into the host's handler (see on_book_update). Only a
//     removal at one of those prices, or the cut of a flagged order there,
//     calls the (b) pass, which reads the queue: the count of orders ahead
//     of each armed order on a removal, and the sizes ahead of an order
//     only after its ratio is met.
//   Decide first.  The pass reads PRE-update estimator state (sigma and tau
//     lag by the current update). Decisions are delivered BEFORE the
//     estimator bookkeeping, so the decays and EWMA updates are not on the
//     path to the callback. A book update for the same symbol fed from
//     inside a callback first completes the deferred bookkeeping, so none is
//     lost or reordered.
//   SIMD.  AVX-512VL on 256-bit registers, one ymm per 64-bit field per
//     quad: compares go straight into k-masks, masked compares chain the
//     conditions, counters are one masked vpaddd (xmm), and the vol test is
//     vcvtqq2pd + two vmulpd + vcmppd. Used when a side has more than
//     kScalarLanes (2) live orders: when a decision fires, the "no fire"
//     prediction fails and the pipeline waits on the fire mask, and a k-mask
//     chain is longer than one predicted compare per live lane (measured
//     8-16 TSC on the fire path with one live order). ymm deliberately: no
//     dependence on 512-bit license or upper-lane power-up transitions on a
//     sporadic hot path. No gathers.
//   Frames.  A function that holds 256-bit values tends to get a realigned
//     stack frame, and one with an addressable local a stack-protector
//     canary, both paid at its entry on every call. So the 256-bit kernels
//     live only in the wide passes, which a symbol with at most two orders a
//     side never calls: its updates go through narrow passes of scalar code
//     (4-8 TSC less per update, measured), and nothing that is inlined into
//     the host's handler holds a 256-bit value.
//   No libm / division / sqrt on the decision path: the floors are one
//     precomputed constant and the spread, the vol test compares
//     depth^2 * den > num * horizon, and its inputs are read lazily, at most
//     once per update, only if some order clears the cheap floors. The
//     variance decay is a second-order expansion (or a range-reduced
//     polynomial for long gaps), after delivery.
//   Reference.  CancelTacticT<false> runs the same arithmetic with scalar
//     code; the test suite runs both and diffs a randomized stream
//     bit-for-bit. Requires IEEE semantics: no -ffast-math for this header.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Host NBBO hook — the only place this header touches the NBBO.
//
// getNBBO(symst_idx): the host system's existing function (defined there,
// not here) returning the best known NBBO for the symbol as one atomic
// two-sided snapshot. Prices in PriceType units; 0 = side absent. A
// one-sided, crossed or absent quote makes the tactic hold (c) for the
// symbol. If the host's function differs in namespace, return type or
// absent-side convention, change this declaration and
// CancelTacticT::read_nbbo() — nothing else reads the NBBO.
// ---------------------------------------------------------------------------
struct Nbbo { PriceType bid; PriceType ask; };
Nbbo getNBBO(uint16_t symst_idx) noexcept;

// getSymbolState(symst_idx): host-defined — the symbol's SymbolState ([2]):
// the cancel tactic reads the round lot there and the join signal the join
// tactic keeps in SymbolState::tac. With this file's MarketDataSystem `mds`:
//   md::SymbolState& md::getSymbolState(uint16_t s) noexcept { return mds.sym(s); }
SymbolState& getSymbolState(uint16_t symst_idx) noexcept;

// ---------------------------------------------------------------------------
// Volatility source — compile-time option.
//
// VolSource::Nbbo (default): sigma is estimated from the NBBO mid, sampled
// at this venue's book updates. VolSource::Venue: from the mid of this
// venue's own ROUND-LOT best bid and offer instead, so odd lots at the top
// of the book do not move it; everything else — the far touch, the
// stability gate, the spread floor, the placement baseline — stays on the
// NBBO. Build with -DMD_CANCEL_VOL_FROM_VENUE=1 to make CancelTactic use it,
// or name it explicitly: CancelTacticT<kCancelTacticSimd, VolSource::Venue>.
//
// getVenueRoundLotBBO(symst_idx): host-defined, used only by
// VolSource::Venue (no link dependency otherwise). This venue's round-lot
// best bid and offer for the symbol AFTER the book update being processed
// has been applied; 0 = no round lot on that side (pauses the estimator).
// Called once per book update of a tracked symbol, after the decision pass
// and its callbacks: its cost is added to every such update but never to
// the path to a cancel. With this file's MarketDataSystem `mds`
// (roundlot_best: the price at which the size from the best level down
// first reaches the symbol's round lot):
//   md::Nbbo md::getVenueRoundLotBBO(uint16_t s) noexcept {
//       const auto b = mds.roundlot_best(s, Side::Buy), a = mds.roundlot_best(s, Side::Sell);
//       return {b ? b->price : 0, a ? a->price : 0};
//   }
// It walks levels until the round lot is reached: about 20 TSC hot with a
// round lot at each best level, 56 with odd lots on the two best levels
// (mdsys_2_cancel_bench part 8).
// ---------------------------------------------------------------------------
enum class VolSource : uint8_t { Nbbo, Venue };
#ifndef MD_CANCEL_VOL_FROM_VENUE
#define MD_CANCEL_VOL_FROM_VENUE 0
#endif
inline constexpr VolSource kCancelVolSource = MD_CANCEL_VOL_FROM_VENUE ? VolSource::Venue : VolSource::Nbbo;
Nbbo getVenueRoundLotBBO(uint16_t symst_idx) noexcept;

// ---------------------------------------------------------------------------
// Size increases — compile-time venue option.
//
// On some venues a Modify that increases an order's size keeps its time
// priority; on others the order goes to the back (a remove and an add). Where
// it keeps priority, the shares ahead of a resting order can grow, so (a) is
// also re-checked when an order ahead of one of ours grows: build with
// -DMD_SIZE_UP_KEEPS_PRIORITY=1 (or name it: CancelTacticT<kSimd, kVol, true>)
// and apply such an increase in place with QBlockList::grow(), whose kQGrew
// the host passes on with the update. Off, kQGrew is ignored.
// ---------------------------------------------------------------------------
#ifndef MD_SIZE_UP_KEEPS_PRIORITY
#define MD_SIZE_UP_KEEPS_PRIORITY 0
#endif
inline constexpr bool kSizeUpKeepsPriority = MD_SIZE_UP_KEEPS_PRIORITY != 0;

// The book updates that trigger a re-check. With each comes the side and
// price of the order the message is about: for Execute the RESTING order
// that traded (so the aggressor is the other side), for Replace the order
// that was replaced. Execute and every removal also feed the queue triggers.
enum class BookEvent : uint8_t { Add, Delete, Modify, Cancel, Replace, Execute };

struct CancelParams {
    // (a) queue position: at placement, again when the shares ahead can
    //     have grown against our size
    uint32_t queue_n                 = 5;                    // N orders ahead; 0 = (a) off
    double   queue_k                 = 3.0;                  // and shares ahead >= K x our size
    // (b) our leading order at a price with no round lot ahead of it, after the
    //     orders ahead cancelled: all of ours at that price
    double   abandon_pct             = 50.0;                 // X: cancelled > X % of the round-lot
                                                             //   orders ahead; < 0 = off
    // (c) too deep
    double   hist_spread_mult        = 1.5;                  // x hist_spread (set_symbol)
    uint32_t min_ticks               = 5;
    uint64_t vol_horizon_cap_ns      = 300'000'000'000ull;   // 5 min
    double   gap_mult                = 10.0;                 // 10 trade events
    double   place_spread_mult       = 1.0;                  // moved-away clause
    // (d) out-competed
    uint32_t aggr_prints_n           = 10;                   // T executions
    bool     passive_side_only       = true;                 // our side only
    bool     aggr_from_signal        = true;                 // an order placed while its side's join
                                                             //   signal lives counts from the signal
    uint32_t aggr_gate_n             = 0;                    // gate, orders ahead; 0 = off
    uint64_t aggr_gate_sz            = 0;                    // gate, shares ahead; 0 = off
    // (c) timing and estimators
    uint64_t far_touch_min_stable_ns = 200'000;              // 200 us
    uint64_t vol_halflife_ns         =  60'000'000'000ull;   // 1 min
    double   gap_alpha               = 0.1;                  // tau EWMA, per event
    uint64_t trade_coalesce_ns       = 0;                    // executions <= this after
                                                             //   the event's first: same event
    double   hysteresis_frac         = 0.7;                  // replace_allowed()
    uint64_t hysteresis_reset_ns     = 300'000'000'000ull;   // forget cancel after
};

#if defined(__AVX512VL__) && defined(__AVX512DQ__)
inline constexpr bool kCancelTacticSimd = true;
#else
inline constexpr bool kCancelTacticSimd = false;
#endif

namespace ct_detail {

// 2^x for x <= 0, x clamped to [-256, 0] (2^-256 is zero for every weight
// used here and keeps all products normal: no denormal assists). Range
// reduction by magic-number rounding, degree-7 polynomial on [-0.5, 0.5]
// (relative-error least-squares fit, constant term pinned at 1 so a zero
// interval decays by exactly 1) in Estrin form, exponent inserted directly.
// Relative error < 1e-10. No libm call, no division, no branch; every
// rounding step is explicit, so the result is bit-identical in every build.
[[gnu::always_inline]] inline double exp2_neg(double x) noexcept {
    constexpr double kMagic = 0x1.8p52;                  // 1.5 * 2^52
    constexpr double c1 = 0x1.62e42ff1998ebp-1, c2 = 0x1.ebfbe079653abp-3,
                     c3 = 0x1.c6b089a4fe724p-5, c4 = 0x1.3b29f24bdd754p-7,
                     c5 = 0x1.5d8b7c9fce583p-10, c6 = 0x1.445d6329f3a96p-13,
                     c7 = 0x1.fce3b6fd9c513p-17;
    x = x < -256.0 ? -256.0 : x;
    const double  t  = x + kMagic;                       // round to nearest integer
    const double  r  = t - kMagic;
    const int64_t ri = std::bit_cast<int64_t>(t) - std::bit_cast<int64_t>(kMagic);
    const double  f  = x - r;                            // [-0.5, 0.5]
    const double  f2 = f * f, f4 = f2 * f2;
    const double  p03 = std::fma(std::fma(c3, f, c2), f2, std::fma(c1, f, 1.0));
    const double  p47 = std::fma(std::fma(c7, f, c6), f2, std::fma(c5, f, c4));
    return std::fma(p47, f4, p03) * std::bit_cast<double>(static_cast<uint64_t>(ri + 1023) << 52);
}

// EWMA weight left after dt: 2^(-dt * inv_halflife). Under 2^-10 half-lives
// (any active name: microseconds against minutes) a second-order expansion,
// error < 6e-11 and ~half the latency; exp2_neg beyond that.
[[gnu::always_inline]] inline double decay(uint64_t dt, double inv_halflife) noexcept {
    constexpr double kLn2 = 0x1.62e42fefa39efp-1;
    const double y = static_cast<double>(dt) * inv_halflife;
    if (y < 0x1p-10) [[likely]] {
        const double z = y * kLn2;
        return std::fma(z, 0.5 * z, 1.0 - z);                // 1 - z + z^2/2
    }
    return exp2_neg(-y);
}

}  // namespace ct_detail

template <bool kSimd, VolSource kVol = kCancelVolSource, bool kSizeUp = kSizeUpKeepsPriority>
class CancelTacticT {
    static_assert(!kSimd || kCancelTacticSimd, "SIMD path needs -mavx512vl -mavx512dq");

public:
    enum class Decision : uint8_t {
        Hold = 0,
        CancelQueue,         // (a) too far back in the queue
        CancelAbandoned,     // (b) our leading order at the price has no round lot ahead, after
                             //     those that were ahead cancelled: every order of ours there
        CancelDepth,         // (c) too deep vs the NBBO far touch
        CancelOutcompeted,   // (d) T executions more aggressive than our price
    };
    using Handle = uint64_t;                         // (generation << 32) | slot
    static constexpr Handle   kInvalid = 0;          // generations start at 1
    static constexpr unsigned kLanes   = 8;          // tracked orders per symbol per side
    static constexpr uint32_t kNoQueue = QBlockList::kBadHandle;   // order not in the book's queues

    struct OrderView {
        uint64_t     id;
        PriceType    px;
        uint32_t     qh;                             // queue handle; kNoQueue: not in the book
        SizeType     sz;                             // remaining, from the queue (0 if not in it)
        uint32_t     queue_ahead;                    // orders ahead in its level, from the queue
        uint64_t     queue_ahead_sz;                 // shares ahead
        uint32_t     n_aggr_prints;                  // (d) executions counted
        uint16_t     ahead_lot_n;                    // (b) round-lot orders ahead at placement
        uint16_t     ahead_cancelled;                // (b) how many of those were cancelled
        FeedTimeType t_place;
        uint16_t     symst;
        Side         side;
        bool         active;
        PriceType    move_min;                       // (c): its depth must exceed this — depth at
                                                     //   placement + place_spread_mult x spread then
    };

    // queues: the book's queue pool (the blocks of every level's QBlockList).
    // The tactic flags tracked slots in it and reads queue positions from it;
    // same thread as the book (single writer).
    CancelTacticT(std::size_t nsymbols, std::size_t max_orders, QBlockPool& queues,
                  CancelParams p = {})
        : syms_(nsymbols), blocks_(2 * nsymbols), sides_(nsymbols, 0),
          track_((nsymbols + 63) / 64, 0), queues_(&queues),
          max_live_(max_orders ? max_orders : 1) {
        assert(nsymbols <= (std::size_t{1} << 14));
        for (Block& b : blocks_)
            for (Quad& q : b.q)
                for (Cold& c : q.cold) c.gen = 1;
        set_params(p);
    }

    // Parameters can change at any time. Turning (b) off disarms it for the
    // orders resting then; they stay disarmed if it is turned on again
    // (rearm(h) re-arms one).
    void set_params(const CancelParams& p) noexcept {
        p_ = p;
        inv_hl_vol_ = 1.0 / static_cast<double>(p.vol_halflife_ns);
        cap_ns_     = static_cast<double>(p.vol_horizon_cap_ns);
        for (Sym& s : syms_) {
            s.floor0 = floor32(static_floor(s));
            if (!abandon_on()) s.barm = 0;
        }
    }
    const CancelParams& params() const noexcept { return p_; }

    // Startup, per symbol: tick and the fixed historical spread, both in
    // price units.
    void set_symbol(uint16_t sym, PriceType tick, PriceType hist_spread) noexcept {
        assert(tick > 0 && hist_spread >= 0);
        Sym& s = syms_[sym];
        s.tick = tick; s.hist_spread = hist_spread;
        s.floor0 = floor32(static_floor(s));
    }

    // ---- per-symbol tracking gate ----------------------------------------------
    // A symbol is tracked while it has active (undecided) orders — kept up to
    // date here as orders are placed, decided, rearmed and released — or while
    // the host pins it, e.g. to keep sigma and tau warm before placing.
    // Unpinning never stops tracking a symbol that still has active orders.
    void set_tracking(uint16_t sym, bool pin) noexcept {
        syms_[sym].pinned = pin;
        retrack(sym);
    }
    [[nodiscard]] bool tracking(uint16_t sym) const noexcept {
        return (track_[sym >> 6] >> (sym & 63u)) & 1u;
    }

    // Warm a tracked symbol's decision state ahead of its update, e.g. at
    // packet decode: the state block plus the order lines of the sides that
    // track orders. Addresses depend only on sym and one byte of a small array.
    // Inlined at every call site, and it has to be: a function that only
    // prefetches has no effect the compiler can see, so GCC deduces it const
    // and DELETES a call to it that it did not inline — silently, at -O1 and
    // above (seen with g++ 13.3; clang 18 inlined it). An untracked symbol
    // costs the one bit test.
    [[gnu::always_inline]] void prefetch(uint16_t sym) const noexcept {
        if (!tracking(sym)) [[likely]] return;
        const char* s = reinterpret_cast<const char*>(&syms_[sym]);
        __builtin_prefetch(s, 1, 3);                // lines 0 and 1: what an update reads. Line 2 is
        __builtin_prefetch(s + 64, 1, 3);           //   only written, when (c) decides
        const uint8_t sd = sides_[sym];
        for (unsigned side = 0; side < 2; ++side) {
            if (!(sd & (1u << side))) continue;
            const Block& b = blocks_[2u * sym + side];
            for (unsigned qi = 0; qi < 2; ++qi) {
                if (qi && !(sd & (4u << side))) break;
                const char* q = reinterpret_cast<const char*>(&b.q[qi]);
                __builtin_prefetch(q, 1, 3);
                __builtin_prefetch(q + 64, 1, 3);
                __builtin_prefetch(q + 128, 0, 3);
            }
            if (abandon_on()) __builtin_prefetch(&b.ab, 1, 3);
        }
    }

    // ---- trigger: one call per venue book update for the symbol --------------
    //   ev      Add / Delete / Modify / Cancel / Replace / Execute
    //   sd, px  side and price of the order the message is about: for
    //           Execute the RESTING order that traded (Buy: a bid was hit)
    //           — its level, which (d) compares with ours; for Replace the
    //           replaced order
    //   ts      the update's feed timestamp
    //   qflags  what the book's QBlockList call for that order returned:
    //           remove() kQGone | kQTracked | kQAhead, reduce() kQTracked |
    //           kQAhead, grow() kQGrew | kQTracked | kQAhead; 0 for an Add
    //           or when no queue was touched
    // After the host applies the update to its book (when the update removed
    // a tracked order, call on_book_remove first). Returns at once for an
    // untracked symbol. F: void(Handle, uint64_t id, Decision), invoked once
    // per decision. With VolSource::Venue the round-lot quote is read after
    // the decision pass and its callbacks, so getVenueRoundLotBBO is never
    // on the path to a cancel; the host's book must still be as this update
    // left it then. If a callback feeds an update for the same symbol, that
    // update completes this one's sample first, from the book as it then
    // stands: the same net move when the two share a timestamp (same-instant
    // changes count as their net), otherwise the later change is dated to
    // this one.
    // Inlined at every call site, as is deliver(): left to the inliner, a
    // host TU can get both out of line (+7 TSC to a cancel, measured), and
    // the untracked return is then a call instead of one bit test. The
    // passes it calls are noinline for the opposite reason: with one call
    // site — a feed handler's usual shape — the inliner pulls them and their
    // 256-bit kernels into the handler, which then realigns its stack and
    // sets up a stack-protector canary on every message it handles (measured
    // through such a handler: 12-14 -> 10 TSC per message of a symbol we
    // have no orders in). The (b) pass is the exception among them: scalar
    // code, and inlined by force — out of line, a removal at one of our
    // prices cost 6 TSC more (measured), and left to the inliner it went
    // either way.
    template <class F>
    [[gnu::always_inline]]
    void on_book_update(uint16_t sym, BookEvent ev, Side sd, PriceType px, FeedTimeType ts,
                        uint8_t qflags, F&& on_cancel) {
        update(sym, ev, sd, px, ts, qflags, on_cancel);
    }

private:
    // on_book_update's work. What a tracked symbol's update does when nothing
    // is decided — the usual case — is kept short: one test says the NBBO is
    // the pair last seen (everything a change entails is behind it); with no
    // live order nothing else is looked at; whether a pass decided is read
    // off `active`, which a decision changes, and only otherwise are the four
    // pending masks tested, in one load.
    template <class F>
    [[gnu::always_inline]]
    void update(uint16_t sym, BookEvent ev, Side sd, PriceType px, FeedTimeType ts, uint8_t qflags,
                F&& on_cancel) {
        if (!tracking(sym)) [[likely]] return;      // no active orders, not pinned: most updates
        Sym& s = syms_[sym];
        const Nbbo q = read_nbbo(sym);
        if (s.bk) [[unlikely]] settle(s, sym);      // fed from a callback: finish that update first
        const FeedTimeType t = advance(s, ts);
        bool moved = false;                         // one test for the usual case: the NBBO as last seen
        if ((q.bid != s.nbb) | (q.ask != s.nbo)) [[unlikely]] { moved = true; nbbo_moved(s, q, t); }
        const bool ex = ev == BookEvent::Execute;
        bool fired = false;
        if (const uint16_t act0 = s.active) {
            if (s.barm && abandon_event(ev, qflags)) {  // (b) is armed on this symbol, and an order left
                const unsigned side = static_cast<unsigned>(sd);   //   or was cut: most removals of a
                Block& b = blocks_[2u * sym + side];               //   symbol we rest in
                const uint32_t arm = (static_cast<uint32_t>(s.barm) >> (side * 8)) & 0xFFu;
                __builtin_prefetch(&b.ab, 0, 3);    // what a removal at one of our prices reads, at
                __builtin_prefetch(b.q[0].t_place, 0, 3);          //   addresses known from sym alone
                if (arm >> 4) __builtin_prefetch(b.q[1].t_place, 0, 3);
                const PriceType pxs = side ? -px : px;
                if (lanes_at(b, arm, pxs)) [[unlikely]] pass_abandon(s, b, arm, pxs, ex, qflags);   // at an armed price
            }
            if (requeue(qflags)) [[unlikely]] pass_queue(s, sym, static_cast<unsigned>(sd), px);
            if (const uint16_t act = s.active) {    // (whatever (a) and (b) have not just decided)
                if (wide(act)) {                    // a live order beyond the first two lanes of a side
                    if (ex) pass_exec_wide(s, sym, static_cast<unsigned>(sd), px, ts, t);
                    else    pass_book_wide(s, sym, t);
                } else {
                    if (ex) pass_exec(s, sym, static_cast<unsigned>(sd), px, ts, t);
                    else    pass_book(s, sym, t);
                }
            }
            fired = s.active != act0;               // a decision takes its order out of `active`: known
        }                                           //   here without reading the pending masks
        // Venue: every update samples the round-lot quote (an unchanged mid is a no-op).
        const bool feed = kVol == VolSource::Nbbo ? moved : true;
        const uint8_t bk = static_cast<uint8_t>((feed ? kBkQuote : 0u) | (ex ? kBkExec : 0u));
        if (fired || pending(s)) [[unlikely]] {     // (pending without fired: decisions an outer update
            s.bk = bk;                              //   of this symbol has yet to deliver)
            deliver(sym, on_cancel);                // estimator work waits for delivery
            if (!s.bk) return;                      // a nested update completed it
            s.bk = 0;
        }
        bookkeep(s, sym, bk, t);
    }

public:
    // ---- own order lifecycle ---------------------------------------------

    // Register one of our resting orders once it is in the book. qh: its
    // handle in its level's QBlockList (what push() returned), which this
    // flags tracked; kNoQueue if the order is not in the book's queues (its
    // queue position is then unknown: (a) and (b) are off for it and the (d)
    // gate passes). Depth and spread at placement — the baseline of (c)'s
    // moved-away clause — come from a fresh getNBBO read: the market the
    // order joins. ts: feed time after which executions count toward (d),
    // e.g. the timestamp of the Add that put it in the book.
    // At placement, with one walk over the queue ahead (only if (a) or (b)
    // is on):
    //   (a) >= queue_n orders and >= queue_k x our size in shares ahead fires
    //       now. Nothing can join ahead of us later, so after this only a
    //       reduction of ours or (kSizeUp) an order ahead growing re-checks it.
    //   (b) counts the market orders ahead holding at least the round lot
    //       (flagging them in the queue, kQAhead, so their departures can be
    //       told apart) and the orders ahead in all; every order of ours has
    //       its own counts, for when it leads its queue.
    //   (d) an order placed while its side's join signal lives starts from
    //       the executions more aggressive than its price since the signal
    //       (SymbolState::tac); at aggr_prints_n it fires now.
    // A decision taken here goes to on_cancel before on_place returns.
    // kInvalid if the symbol side already tracks kLanes orders or max_orders
    // are live. A call, never inlined: the queue walk and the join seed are
    // 256-bit code, which must not end up in the host's message handler (see
    // on_book_update).
    template <class F>
    [[gnu::noinline]] Handle on_place(uint64_t id, uint16_t sym, Side sd, PriceType px, FeedTimeType ts,
                                      uint32_t qh, F&& on_cancel) {
        Sym& s = syms_[sym];
        const unsigned side = static_cast<unsigned>(sd);
        const uint32_t used = (static_cast<uint32_t>(s.used) >> (side * 8)) & 0xFFu;
        if (used == 0xFFu || live_ >= max_live_) [[unlikely]] { ++place_failures_; return kInvalid; }
        const Nbbo nb = read_nbbo(sym);
        const unsigned lane = static_cast<unsigned>(std::countr_zero(~used & 0xFFu));
        Quad& q = blocks_[2u * sym + side].q[lane >> 2];
        const unsigned j = lane & 3u;
        const PriceType pxs = side ? -px : px;                         // sells negated
        PriceType move_min = 0;                                        // no NBBO: fail toward cancel
        if (quote_ok(nb.bid, nb.ask))
            move_min = ((side ? -nb.bid : nb.ask) - pxs) +
                       static_cast<PriceType>(p_.place_spread_mult * static_cast<double>(nb.ask - nb.bid));
        bool fire_a = false;                                           // (a), (b): the queue ahead
        uint32_t na = 0, n0 = 0;
        const SymbolState& st = getSymbolState(sym);
        s.lot = st.round_lot;
        if (qh != kNoQueue && (queue_on() || abandon_on())) {
            const QScan a = QBlockList::scan_ahead(*queues_, qh, st.round_lot, abandon_on());
            fire_a = queue_on() && queue_full(a.n, a.sz, QBlockList::size_of(*queues_, qh));
            if (abandon_on()) { na = a.n; n0 = std::min<uint32_t>(a.n_lot, 0xFFFFu); }
        }
        const uint32_t n_aggr = p_.aggr_from_signal                    // (d) from the join signal
                              ? join_aggr(st.tac, side, px, p_.passive_side_only) : 0u;
        q.px[j] = pxs; q.move_min[j] = move_min; q.t_place[j] = ts;
        q.n_aggr[j] = n_aggr; q.qh[j] = qh;
        Cold& c = q.cold[j];
        c.id = id;
        BLine& ab = blocks_[2u * sym + side].ab;
        ab.na[lane] = na; ab.n0[lane] = static_cast<uint16_t>(n0); ab.nc[lane] = 0;
        if (qh != kNoQueue) QBlockList::track(*queues_, qh, true);
        const uint16_t bit = static_cast<uint16_t>(1u << (side * 8 + lane));
        s.used = static_cast<uint16_t>(s.used | bit);
        s.active = static_cast<uint16_t>(s.active | bit);
        if (n0) s.barm = static_cast<uint16_t>(s.barm | bit);
        sides_[sym] = side_bits(s.used);
        ++live_;
        const Handle h = make_handle(c.gen, slot_of(sym, side, lane));
        const bool fire_d = !fire_a && n_aggr != 0 && n_aggr >= p_.aggr_prints_n && aggr_gate(qh);
        if (fire_a || fire_d) [[unlikely]] {
            s.active = static_cast<uint16_t>(s.active & ~bit);
            s.barm   = static_cast<uint16_t>(s.barm & ~bit);
            retrack(sym);
            on_cancel(h, id, fire_a ? Decision::CancelQueue : Decision::CancelOutcompeted);
            return h;
        }
        retrack(sym);
        return h;
    }

    // ---- an order recognised as ours after it entered the book ---------------
    //
    // on_place assumes the host knows an order is ours when its Add arrives.
    // The host learns it from the exchange's acknowledgement, which carries
    // the order's id: when the ack comes first, it calls on_place as the Add
    // arrives and nothing below applies. When the Add came first, it calls
    // on_place_late at the ack, with the order's queue handle (from its order
    // table) and the Add's feed timestamp, t_add. The order is tracked from
    // the call, the queue and the NBBO as they stand now taking the place of
    // what they were at the Add:
    //   (a)  judged on the queue ahead now — as at the Add but for the orders
    //        that have left since: only shallower.
    //   (b)  counts the round lots ahead now, and their cancellations from
    //        now. What left between the Add and the call is not seen: a
    //        cancellation then does not count, and a fill then drops out of
    //        the count it was part of, so (b) can come later or earlier than
    //        it would have. An order with no round lot ahead at the call has
    //        no counts: it never gets (b), and while it leads its queue none
    //        of ours behind it does either.
    //   (c)  its moved-away baseline is the NBBO now.
    //   (d)  with aggr_from_signal, starts from the executions more aggressive
    //        than its price since its side's join signal, if that still lives
    //        — the signal the order answered, so the strategy kills it after
    //        this call, not before; otherwise from the call. Executions after
    //        t_add count from then on.
    // Two corrections, made here only. An order of ours behind it at its
    // price that counted it as a market round lot ahead no longer does (with
    // MD_SIZE_UP_KEEPS_PRIORITY approximately: an order that grew into a
    // round lot may be taken out of the count of one of ours that never
    // counted it). And if the side's live join signal was raised by this
    // order itself — our round lot improving the price, signalled before we
    // knew it was ours: the signal is at its price, from an Add with its
    // timestamp, and nothing rests ahead of it — the signal's kill flag is
    // set, after (d) has started from it (since the signal is then since the
    // Add); LateInfo says so, and the join tactic ends it at its next read.
    // A decision taken here goes to on_cancel before the call returns.
    // kInvalid as on_place. The order must be in the book at qh when called:
    // the host registers no order that left before its ack. Called again for
    // an order already tracked, it returns that order's handle. About what
    // on_place costs, however late the call; a call, never inlined, as
    // on_place.
    struct LateInfo {
        bool own_signal = false;               // the side's join signal was this order's: its kill flag is set
    };

    template <class F>
    [[gnu::noinline]] Handle on_place_late(uint64_t id, uint16_t sym, Side sd, PriceType px, FeedTimeType t_add,
                                           uint32_t qh, F&& on_cancel, LateInfo* info = nullptr) {
        if (info) *info = LateInfo{};
        if (qh != kNoQueue && QBlockList::tracked(*queues_, qh)) return find(sym, sd, qh);   // told twice
        Sym& s = syms_[sym];
        const unsigned side = static_cast<unsigned>(sd);
        const uint32_t used = (static_cast<uint32_t>(s.used) >> (side * 8)) & 0xFFu;
        if (used == 0xFFu || live_ >= max_live_) [[unlikely]] { ++place_failures_; return kInvalid; }
        ++late_placed_;
        SymbolState& st = getSymbolState(sym);
        bool own = false;
        if (qh != kNoQueue) {
            const PriceType pxs = side ? -px : px;
            Block& blk = blocks_[2u * sym + side];
            const uint32_t my_n = QBlockList::ahead_n(*queues_, qh);
            uint32_t behind = 0;                                       // ours behind it at its price
            for (uint32_t m = used; m; m &= m - 1) {
                const unsigned l = static_cast<unsigned>(std::countr_zero(m));
                const Quad& q = blk.q[l >> 2];
                if (q.px[l & 3u] == pxs && in_book(q.qh[l & 3u]) && QBlockList::ahead_n(*queues_, q.qh[l & 3u]) > my_n)
                    behind |= 1u << l;
            }
            uncount(s, blk, side, qh, behind);
            if (my_n == 0)
                if (const JoinSide* js = join_live(st.tac, side)) own = js->px == px && js->t == t_add;
        }
        const Handle h = on_place(id, sym, sd, px, t_add, qh, on_cancel);
        if (own) {
            __atomic_store_n(&st.tac.side[side].kill, uint8_t{1}, __ATOMIC_RELAXED);
            if (info) info->own_signal = true;
        }
        return h;
    }

    // A tracked order left the book — QBlockList::remove() reported
    // kQTracked: filled, canceled or replaced. Releases its lane. Call it before
    // on_book_update for the same message, so the pass cannot decide on an
    // order that is already gone. False if no live order of this symbol side
    // has that queue handle (e.g. on_done came first).
    bool on_book_remove(uint16_t sym, Side sd, uint32_t qh) noexcept {
        return on_done(find(sym, sd, qh));
    }

    // Handle of the live order of this symbol side whose queue handle is qh.
    [[nodiscard]] Handle find(uint16_t sym, Side sd, uint32_t qh) const noexcept {
        if (qh == kNoQueue) return kInvalid;
        const unsigned side = static_cast<unsigned>(sd);
        const Block& b = blocks_[2u * sym + side];
        for (uint32_t m = (static_cast<uint32_t>(syms_[sym].used) >> (side * 8)) & 0xFFu; m; m &= m - 1) {
            const unsigned l = static_cast<unsigned>(std::countr_zero(m));
            const Quad& q = b.q[l >> 2];
            if (q.qh[l & 3u] == qh) return make_handle(q.cold[l & 3u].gen, slot_of(sym, side, l));
        }
        return kInvalid;
    }

    // Cancel rejected and the order is still live: let the tactic fire again.
    // (b) re-fires at the next removal at its level (or cut of a flagged
    // order there) while it still holds for our leading order there — if that
    // is this order, or once that one is rearmed too: a (b) decision takes
    // every live order of ours at the price, so after rejects each is
    // rearmed on its own. (c) fires
    // as soon as its condition is next evaluated true. For (d) the
    // execution count is pulled back to one below the threshold, so it takes
    // one more counted execution to re-fire rather than every one from here
    // on (a counter that stays saturated would turn one reject into a cancel
    // request per execution); its queue gate is re-evaluated then. (a) fires
    // again at its next re-check (one of ours at that price reduced, or with
    // kSizeUp an order there grown); evaluate(h) re-checks it on the queue as
    // it is now.
    bool rearm(Handle h) noexcept {
        const Ref r = resolve(h);
        if (!r.s) return false;
        uint32_t& n = r.q->n_aggr[r.j];
        if (n >= p_.aggr_prints_n) n = p_.aggr_prints_n - 1;
        const uint16_t bit = static_cast<uint16_t>(1u << r.bit);
        r.s->active = static_cast<uint16_t>(r.s->active | bit);
        const unsigned lane = r.bit & 7u;
        if (abandon_on() && r.b->ab.n0[lane]) {     // (b): departures while decided were not seen
            r.s->barm = static_cast<uint16_t>(r.s->barm | bit);
            if (in_book(r.q->qh[r.j])) r.b->ab.na[lane] = QBlockList::ahead_n(*queues_, r.q->qh[r.j]);
        }
        retrack(sym_of(r));
        return true;
    }

    // Stop tracking this order (cancel acked, or the host is done with it):
    // releases its lane. The book's tracked flag is left to the book: it
    // describes the order in the slot, and remove() clears it.
    bool on_done(Handle h) noexcept {
        const Ref r = resolve(h);
        if (!r.s) return false;
        release(r);
        return true;
    }

    // ---- queries ---------------------------------------------------------

    [[nodiscard, gnu::noinline]] std::optional<OrderView> order(Handle h) const noexcept {
        const Ref r = resolve(h);
        if (!r.s) return std::nullopt;
        const Quad& q = *r.q;
        const unsigned j = r.j, side = r.bit >> 3;
        QAhead a{0, 0};
        SizeType sz = 0;
        if (in_book(q.qh[j])) {
            a  = QBlockList::ahead_of(*queues_, q.qh[j]);
            sz = QBlockList::size_of(*queues_, q.qh[j]);
        }
        return OrderView{q.cold[j].id, side ? -q.px[j] : q.px[j], q.qh[j], sz, a.n, a.sz,
                         q.n_aggr[j], r.b->ab.n0[r.bit & 7u], r.b->ab.nc[r.bit & 7u], q.t_place[j], sym_of(r),
                         static_cast<Side>(side), ((r.s->active >> r.bit) & 1u) != 0, q.move_min[j]};
    }

    // What a book update would decide for this order on the state as of the
    // symbol's last book update (NBBO and clock as seen then; no NBBO read,
    // no state change). Like order(), a call: both read the queue with
    // 256-bit code, and a host may well use them inside its cancel callback.
    [[nodiscard, gnu::noinline]] Decision evaluate(Handle h) const noexcept {
        const Ref r = resolve(h);
        if (!r.s) return Decision::Hold;
        const Sym& s = *r.s;
        const Quad& q = *r.q;
        const uint32_t qh = q.qh[r.j];
        if (queue_on() && in_book(qh)) {
            const QAhead a = QBlockList::ahead_of(*queues_, qh);
            if (queue_full(a.n, a.sz, QBlockList::size_of(*queues_, qh))) return Decision::CancelQueue;
        }
        const unsigned side = r.bit >> 3, lane = r.bit & 7u;
        if (abandon_on()) {                         // (b): for it, or for our leading order at its price
            if (abandoned(s, *r.b, lane, qh)) return Decision::CancelAbandoned;
            const uint32_t arm = (static_cast<uint32_t>(s.barm) >> (side * 8)) & 0xFFu & ~(1u << lane);
            for (uint32_t m = arm; m; m &= m - 1) {
                const unsigned l = static_cast<unsigned>(std::countr_zero(m));
                const Quad& o = r.b->q[l >> 2];
                if (o.px[l & 3u] == q.px[r.j] && abandoned(s, *r.b, l, o.qh[l & 3u])) return Decision::CancelAbandoned;
            }
        }
        if (q.n_aggr[r.j] >= p_.aggr_prints_n && aggr_gate(qh))
            return Decision::CancelOutcompeted;
        if (!quote_ok(s) || !far_stable(s, side, s.t_now)) return Decision::Hold;
        Floors f = floors(s);
        return block_c_ref(*r.b, 1u << lane, far_of(s, side), f, s) ? Decision::CancelDepth
                                                                    : Decision::Hold;
    }

    // Re-entry hysteresis after a depth cancel on this symbol/side: allow a
    // new order at px only once its depth (against a fresh getNBBO read) is
    // back under hysteresis_frac of the threshold that fired, or the cancel
    // has aged out.
    [[nodiscard]] bool replace_allowed(uint16_t sym, Side sd, PriceType px,
                                       FeedTimeType ts) const noexcept {
        const Sym& s = syms_[sym];
        const unsigned k = static_cast<unsigned>(sd);
        if (s.last_cancel_thr[k] == 0) return true;
        if (elapsed(std::max(ts, s.t_now), s.t_last_cancel[k]) > p_.hysteresis_reset_ns) return true;
        const Nbbo nb = read_nbbo(sym);
        if (!quote_ok(nb.bid, nb.ask)) return false;
        const PriceType depth = (sd == Side::Buy) ? nb.ask - px : px - nb.bid;
        return static_cast<double>(depth) <= p_.hysteresis_frac *
                                              static_cast<double>(s.last_cancel_thr[k]);
    }

    // Current (c) threshold for a symbol, in price units, as of its last update.
    [[nodiscard]] PriceType threshold(uint16_t sym) const noexcept {
        const Sym& s = syms_[sym];
        Floors f = floors(s);
        ensure_full(f, s);
        return full_threshold(f);
    }
    [[nodiscard]] double sigma_rate(uint16_t sym) const noexcept {          // price / sqrt(ns)
        const Sym& s = syms_[sym];
        return s.var_den > 0.0 ? std::sqrt(s.var_num / s.var_den) : 0.0;
    }
    [[nodiscard]] double trade_gap_ns(uint16_t sym) const noexcept {
        const Sym& s = syms_[sym];
        return s.gap_seeded ? s.gap_ewma_ns : 0.0;
    }
    [[nodiscard]] FeedTimeType sym_time(uint16_t sym)   const noexcept { return syms_[sym].t_now; }
    [[nodiscard]] std::size_t  live_orders()            const noexcept { return live_; }
    [[nodiscard]] std::size_t  place_failures()         const noexcept { return place_failures_; }
    // Orders registered by on_place_late.
    [[nodiscard]] std::size_t  late_placed()            const noexcept { return late_placed_; }

private:
    // ---- state ---------------------------------------------------------------

    static constexpr uint8_t kBkQuote = 1, kBkExec = 2;   // Sym::bk

    struct alignas(64) Sym {
        // line 0 — read first by every update
        PriceType    nbb = 0, nbo = 0;               // NBBO as last read by an update
        FeedTimeType t_now = 0;                      // symbol clock: max update timestamp
        FeedTimeType t_bid_chg = 0, t_ask_chg = 0;   // last observed change of each side
        int32_t      floor0 = 0;                     // max(min_ticks*tick, hist_mult*hist_spread)
        uint32_t     lot = 0;                        // the symbol's round lot, as of our last placement: (b)
        uint16_t     used = 0, active = 0;           // bit = side * 8 + lane
        uint16_t     pend_a = 0, pend_b = 0, pend_c = 0, pend_d = 0;   // decided by (a)-(d),
                                                                      //   not yet delivered
        uint16_t     barm = 0;                       // (b) armed: active, round-lot orders were ahead
        uint8_t      bk = 0;                         // estimator work deferred past delivery
        bool         pinned     : 1 = false;         // set_tracking()
        bool         mid_init   : 1 = false;
        bool         burst_open : 1 = false;
        bool         have_exec  : 1 = false;
        bool         gap_seeded : 1 = false;
        // line 1 — sigma and tau
        alignas(64) double var_num = 0.0;            // sum w*dmid^2
        double       var_den = 0.0;                  // sum w*dt
        double       mid_prev = 0.0, mid_anchor = 0.0, last_inc = 0.0;
        FeedTimeType t_mid = 0;
        double       gap_ewma_ns = std::numeric_limits<double>::infinity();   // tau; inf -> cap
        FeedTimeType t_event = 0;                    // first execution of the current event
        // line 2 — re-entry hysteresis (written when (c) fires), configuration
        alignas(64) PriceType last_cancel_thr[2] = {0, 0};
        FeedTimeType t_last_cancel[2] = {0, 0};
        PriceType    tick = 100, hist_spread = 0;
    };
    static_assert(sizeof(Sym) == 192 && offsetof(Sym, var_num) == 64 &&
                  offsetof(Sym, last_cancel_thr) == 128, "Sym line layout");
    static_assert(offsetof(Sym, pend_d) == offsetof(Sym, pend_a) + 6, "the four pending masks are one word");

    // Any decision pending delivery: the four masks in one load. Not to be
    // read straight after a pass has stored to one of them — a wide load over
    // a narrow store just made does not forward — which is why update()
    // first asks whether `active` changed.
    static uint64_t pending(const Sym& s) noexcept {
        uint64_t v;
        std::memcpy(&v, reinterpret_cast<const char*>(&s) + offsetof(Sym, pend_a), sizeof v);
        return v;
    }

    struct Cold {                              // one lane, 16 B
        uint64_t id;
        uint32_t gen;
        uint32_t pad_;
    };

    struct alignas(64) Quad {                  // 4 orders of one side
        PriceType    px[4];                    // line 0 — sells negated: depth = far - px
        PriceType    move_min[4];              //          (c) needs depth > move_min
        FeedTimeType t_place[4];               // line 1
        uint32_t     n_aggr[4];
        uint32_t     qh[4];                    //          queue handle in the book
        Cold         cold[4];                  // line 2 — read when firing
    };
    static_assert(sizeof(Quad) == 192 && offsetof(Quad, move_min) == 32 &&
                  offsetof(Quad, t_place) == 64 && offsetof(Quad, n_aggr) == 96 &&
                  offsetof(Quad, qh) == 112 && offsetof(Quad, cold) == 128, "Quad line layout");

    struct alignas(64) BLine {                 // (b), per lane: read when an order leaves an armed price
        uint32_t na[8];                        // orders ahead of ours, as of the last removal seen
        uint16_t n0[8];                        // round-lot market orders ahead at placement (flagged)
        uint16_t nc[8];                        // how many of those left by cancellation
    };
    static_assert(sizeof(BLine) == 64, "one line");

    struct Block { Quad q[2]; BLine ab; };     // lanes 0-3, 4-7; their (b) state

    struct Floors {                            // per-update (c) context
        PriceType cheap  = 0;                  // max(floor0, spread_now)
        double    den    = 1.0;                // vol test: d*d*den > numhor          (lazy;
        double    numhor = -1.0;               //   1 and -1 = no vol yet: passes)
        bool      full   = false;
    };

    struct Ref {                               // a resolved handle
        Sym*     s = nullptr;
        Block*   b = nullptr;
        Quad*    q = nullptr;
        unsigned bit = 0, j = 0;               // side * 8 + lane; lane within quad
    };

    // The NBBO this update read is not the pair last seen: stamp each side
    // that moved; keep the new pair.
    [[gnu::always_inline]] static void nbbo_moved(Sym& s, const Nbbo& q, FeedTimeType t) noexcept {
        if (q.bid != s.nbb) s.t_bid_chg = t;
        if (q.ask != s.nbo) s.t_ask_chg = t;
        s.nbb = q.bid; s.nbo = q.ask;
    }

    // The order at qh turns out to be ours. If one of ours behind it flagged
    // it as a market round lot ahead, each of ours behind it at that price
    // counted it (sizes do not grow: once flagged, it stayed counted). Take it
    // out of their counts and clear the flag.
    void uncount(Sym& s, Block& blk, unsigned side, uint32_t qh, uint32_t behind) noexcept {
        if (!QBlockList::flagged_ahead(*queues_, qh)) return;
        QBlockList::flag_ahead(*queues_, qh, false);
        for (uint32_t m = behind; m; m &= m - 1) {
            const unsigned l = static_cast<unsigned>(std::countr_zero(m));
            if (!blk.ab.n0[l]) continue;
            if (--blk.ab.n0[l] < blk.ab.nc[l]) blk.ab.nc[l] = blk.ab.n0[l];
            if (!blk.ab.n0[l]) s.barm = static_cast<uint16_t>(s.barm & ~(1u << (side * 8 + l)));
        }
    }

    // ---- small helpers -------------------------------------------------------

    // The one NBBO read (see getNBBO). Adapt here if the host's snapshot type
    // or absent-side convention differs.
    static Nbbo read_nbbo(uint16_t sym) noexcept {
        const auto q = getNBBO(sym);
        return Nbbo{q.bid, q.ask};
    }
    // This venue's round-lot BBO, for the volatility only (VolSource::Venue).
    static Nbbo read_venue_bbo(uint16_t sym) noexcept {
        const auto q = getVenueRoundLotBBO(sym);
        return Nbbo{q.bid, q.ask};
    }

    PriceType static_floor(const Sym& s) const noexcept {
        return std::max(static_cast<PriceType>(p_.min_ticks) * s.tick,
                        static_cast<PriceType>(p_.hist_spread_mult * static_cast<double>(s.hist_spread)));
    }
    // Kept in 32 bits (line 0 of Sym is full): a floor beyond 2^31 price units is no floor.
    static int32_t floor32(PriceType f) noexcept {
        return static_cast<int32_t>(std::min<PriceType>(f, std::numeric_limits<int32_t>::max()));
    }
    static FeedTimeType elapsed(FeedTimeType later, FeedTimeType earlier) noexcept {
        return later > earlier ? later - earlier : 0;
    }
    static FeedTimeType advance(Sym& s, FeedTimeType ts) noexcept {
        if (ts > s.t_now) s.t_now = ts;
        return s.t_now;
    }
    static bool quote_ok(PriceType bid, PriceType ask) noexcept { return bid > 0 && bid <= ask; }
    static bool quote_ok(const Sym& s) noexcept { return quote_ok(s.nbb, s.nbo); }
    static PriceType far_of(const Sym& s, unsigned side) noexcept { return side ? -s.nbb : s.nbo; }
    bool far_stable(const Sym& s, unsigned side, FeedTimeType t) const noexcept {
        return elapsed(t, side ? s.t_bid_chg : s.t_ask_chg) >= p_.far_touch_min_stable_ns;
    }
    static uint8_t side_bits(uint32_t used) noexcept {          // see sides_
        return static_cast<uint8_t>(((used & 0xFFu) ? 1u : 0u) | ((used >> 8) ? 2u : 0u) |
                                    ((used & 0xF0u) ? 4u : 0u) | ((used & 0xF000u) ? 8u : 0u));
    }
    static uint32_t slot_of(uint16_t sym, unsigned side, unsigned lane) noexcept {
        return ((2u * sym + side) << 3) | lane;
    }
    static Handle make_handle(uint32_t gen, uint32_t slot) noexcept {
        return (static_cast<Handle>(gen) << 32) | slot;
    }
    uint16_t sym_of(const Ref& r) const noexcept { return static_cast<uint16_t>(r.s - syms_.data()); }

    // Recompute the symbol's tracking bit. Resuming (off -> on) restarts the
    // trade-gap measurement: executions were not seen while untracked.
    void retrack(uint16_t sym) noexcept {
        Sym& s = syms_[sym];
        uint64_t& w = track_[sym >> 6];
        const uint64_t bit = 1ull << (sym & 63u);
        if (s.active != 0 || s.pinned) {
            if (!(w & bit)) { w |= bit; s.have_exec = false; }
        } else {
            w &= ~bit;
        }
    }

    // The order's slot still holds a tracked order (trk is a subset of occ,
    // and remove() clears it): a lane whose order left the book without
    // on_book_remove() reads as "not in the book", never as another order's
    // position.
    bool in_book(uint32_t qh) const noexcept {
        return qh != kNoQueue && QBlockList::tracked(*queues_, qh);
    }

    Ref resolve(Handle h) noexcept {
        const uint32_t slot = static_cast<uint32_t>(h), bi = slot >> 3, lane = slot & 7u;
        if (bi >= blocks_.size()) return {};
        Sym& s = syms_[bi >> 1];
        Block& b = blocks_[bi];
        Quad& q = b.q[lane >> 2];
        const unsigned bit = (bi & 1u) * 8 + lane, j = lane & 3u;
        if (!((s.used >> bit) & 1u) || q.cold[j].gen != static_cast<uint32_t>(h >> 32)) return {};
        return {&s, &b, &q, bit, j};
    }
    Ref resolve(Handle h) const noexcept { return const_cast<CancelTacticT*>(this)->resolve(h); }

    void release(const Ref& r) noexcept {
        const uint16_t bit = static_cast<uint16_t>(1u << r.bit), keep = static_cast<uint16_t>(~bit);
        Sym& s = *r.s;
        s.used &= keep;
        // Each other mask is written only if it holds the lane, in a branch of
        // its own. The usual caller is the callback, after delivery: the
        // order is decided, so none of them does. Unconditional, GCC merges
        // these into wide loads across fields the decision path has just
        // stored one by one (mark_*, deliver) — a store-forwarding stall,
        // measured as a +150-300 TSC tail on 1-2% of decided calls.
        if (s.active & bit) s.active &= keep;
        if (s.barm & bit)   s.barm &= keep;
        if ((s.pend_a | s.pend_b | s.pend_c | s.pend_d) & bit) [[unlikely]] {
            s.pend_a &= keep; s.pend_b &= keep; s.pend_c &= keep; s.pend_d &= keep;
        }
        const uint16_t sym = sym_of(r);
        sides_[sym] = side_bits(s.used);
        uint32_t& gen = r.q->cold[r.j].gen;
        if (++gen == 0) gen = 1;                                        // handle 0 stays invalid
        --live_;
        retrack(sym);
    }

    // ---- decision passes -------------------------------------------------------

    // The lanes among `cand` (one side's eight) whose order rests at price
    // pxs. On every removal of a symbol with (b) armed, so: compares at
    // addresses known from sym and side alone — the loads leave with the Sym
    // line's, not after it — and nothing per lane unless the price matches.
    [[gnu::always_inline]] uint32_t lanes_at(const Block& b, uint32_t cand, PriceType pxs) const noexcept {
        if constexpr (kSimd) {
            return lanes_at_simd(b, cand, pxs);
        } else {
            uint32_t m = 0;
            for (uint32_t c = cand; c; c &= c - 1) {
                const unsigned l = static_cast<unsigned>(std::countr_zero(c));
                m |= static_cast<uint32_t>(b.q[l >> 2].px[l & 3u] == pxs) << l;
            }
            return m;
        }
    }

    // The updates that can make (b) true for an order of ours: one that
    // removes an order (the last round lot ahead of ours leaving; a counted
    // one cancelling), and one that cuts a market order in place (the last
    // round lot ahead of ours worked down to an odd lot). Of the cuts, only a
    // flagged order's matter: a market order ahead of ours holding a round
    // lot was flagged when ours was placed — unless (kSizeUp) it has grown
    // into one since, so there every cut that is not of ours counts. An Add,
    // a size increase, a change to one of ours: never.
    static bool abandon_event(BookEvent ev, uint8_t qflags) noexcept {
        if constexpr (kSizeUp)
            return (qflags & kQGone) || (ev != BookEvent::Add && !(qflags & (kQTracked | kQGrew)));
        else
            return (qflags & kQGone) || (qflags & (kQTracked | kQAhead | kQGrew)) == kQAhead;
    }

    // An order left the queue at price pxs (kQGone), or a market order there
    // was cut, where one of our armed orders `arm` (lanes of block b) rests:
    // (b). On a removal, each armed order's count of orders ahead says
    // whether the order that left was ahead of it (it dropped by one); if so,
    // and that order was a flagged market order (kQAhead, not kQTracked) that
    // left by anything but an execution, it was a cancellation. (b) holds for
    // an order once its cancellations exceed abandon_pct % of the round-lot
    // orders that were ahead of it, and nothing ahead of it is one of ours or
    // still holds a round lot — it is our leading order at the price, at the
    // front or behind odd lots only. Then every live order of ours at the
    // price is decided with it. The ratio is tested first: until it is met
    // the queue's sizes are not read. The lanes are matched again here, with
    // a predicted compare each, rather than taken from the caller's mask: the
    // queue reads then start from the armed mask and do not wait for the
    // vector compare (measured: 8 TSC).
    [[gnu::always_inline]] void pass_abandon(Sym& s, Block& b, uint32_t arm, PriceType pxs, bool ex,
                                             uint8_t qflags) noexcept {
        const bool gone = (qflags & kQGone) != 0;
        const bool cancelled = (qflags & (kQGone | kQAhead | kQTracked)) == (kQGone | kQAhead) && !ex;
        bool fire = false;
        for (uint32_t m = arm; m; m &= m - 1) {
            const unsigned l = static_cast<unsigned>(std::countr_zero(m)), j = l & 3u;
            const Quad& q = b.q[l >> 2];
            if (q.px[j] != pxs || !in_book(q.qh[j])) continue;
            bool front = false;                                       // nothing at all ahead of it
            if (gone) {
                const uint32_t n = QBlockList::ahead_n(*queues_, q.qh[j]);
                if (n < b.ab.na[l]) {                                 // it was ahead of ours
                    b.ab.na[l] = n;
                    if (cancelled && b.ab.nc[l] < b.ab.n0[l]) ++b.ab.nc[l];
                }
                front = n == 0;
            }
            if (ratio_met(b.ab.nc[l], b.ab.n0[l]) && (front || QBlockList::clear_ahead(*queues_, q.qh[j], s.lot)))
                fire = true;
        }
        if (fire) [[unlikely]] {
            const uint32_t bi = static_cast<uint32_t>(&b - blocks_.data());
            fire_level(s, static_cast<uint16_t>(bi >> 1), bi & 1u, b, pxs);
        }
    }

    // (b) holds for our leading order at price pxs: decide every live order
    // of ours there — armed or not, in the book's queues or not.
    [[gnu::always_inline]] void fire_level(Sym& s, uint16_t sym, unsigned side, const Block& b,
                                           PriceType pxs) noexcept {
        const uint32_t act = (static_cast<uint32_t>(s.active) >> (side * 8)) & 0xFFu;
        mark_b(s, sym, side, lanes_at(b, act, pxs));
    }

    // (a) again, for our active orders at price px on side `side`, when the
    // shares ahead can have grown against our size: one of ours there was
    // reduced, or (kSizeUp) an order there grew in place — any order: one of
    // ours growing puts more shares ahead of ours behind it.
    static bool requeue(uint8_t qflags) noexcept {
        constexpr uint8_t k = kQGone | kQTracked | kQGrew;
        return (qflags & k) == kQTracked || (kSizeUp && (qflags & (kQGone | kQGrew)) == kQGrew);
    }
    [[gnu::noinline]] void pass_queue(Sym& s, uint16_t sym, unsigned side, PriceType px) noexcept {
        const uint32_t act = (static_cast<uint32_t>(s.active) >> (side * 8)) & 0xFFu;
        if (!act || !queue_on()) return;
        const PriceType pxs = side ? -px : px;
        const Block& b = blocks_[2u * sym + side];
        uint32_t fire = 0;
        for (uint32_t m = act; m; m &= m - 1) {
            const unsigned l = static_cast<unsigned>(std::countr_zero(m)), j = l & 3u;
            const Quad& q = b.q[l >> 2];
            if (q.px[j] != pxs || !in_book(q.qh[j])) continue;
            const QAhead a = QBlockList::ahead_of(*queues_, q.qh[j]);
            if (queue_full(a.n, a.sz, QBlockList::size_of(*queues_, q.qh[j]))) fire |= 1u << l;
        }
        if (fire) [[unlikely]] mark_a(s, sym, side, fire);
    }

    // Any non-execution update: (c) on every live order of the symbol.
    // Each pass exists twice. The narrow one runs when every live order of
    // the symbol sits in the first two lanes of its side — any symbol with at
    // most two orders a side, since an order takes the lowest free lane (one
    // whose cancel is in flight holds its lane until released) — and is
    // scalar code only: no 256-bit value, so no realigned frame, no
    // stack-protector canary, fewer registers saved. The wide one holds the
    // 256-bit kernels. Which runs changes nothing a decision depends on: the
    // scalar kernels are the reference the SIMD ones are tested against, and
    // a side with at most kScalarLanes live orders used them already.
    // All four stay out of line (noinline): inlined into the host's handler
    // through on_book_update, the wide ones cost that function a realigned
    // stack frame on every message it handles, an untracked symbol's included
    // (measured: 2-4 TSC), and the narrow ones gain nothing by it (measured).
    [[gnu::noinline]] void pass_book(Sym& s, uint16_t sym, FeedTimeType t)      { book_pass<false>(s, sym, t); }
    [[gnu::noinline]] void pass_book_wide(Sym& s, uint16_t sym, FeedTimeType t) { book_pass<true>(s, sym, t); }
    template <bool kWide>
    [[gnu::always_inline]] void book_pass(Sym& s, uint16_t sym, FeedTimeType t) {
        if (!quote_ok(s)) return;                                      // degenerate NBBO: hold
        Floors f = floors(s);
        for (unsigned side = 0; side < 2; ++side) {
            const uint32_t act = (static_cast<uint32_t>(s.active) >> (side * 8)) & 0xFFu;
            if (!act || !far_stable(s, side, t)) continue;
            const Block& b = blocks_[2u * sym + side];
            prefetch_quads(b, act, false);
            const uint32_t m = block_c<kWide>(b, act, far_of(s, side), f, s);
            if (m) [[unlikely]] mark_c(s, sym, side, m, t, threshold_of(f, s));
        }
    }

    // Execution against a resting order on side `rest` at price px: (d) on
    // the sides it counts for, then (c) on the orders (d) did not take.
    [[gnu::noinline]] void pass_exec(Sym& s, uint16_t sym, unsigned rest, PriceType px, FeedTimeType ts,
                                     FeedTimeType t) { exec_pass<false>(s, sym, rest, px, ts, t); }
    [[gnu::noinline]] void pass_exec_wide(Sym& s, uint16_t sym, unsigned rest, PriceType px, FeedTimeType ts,
                                          FeedTimeType t) { exec_pass<true>(s, sym, rest, px, ts, t); }
    template <bool kWide>
    [[gnu::always_inline]] void exec_pass(Sym& s, uint16_t sym, unsigned rest, PriceType px, FeedTimeType ts,
                                          FeedTimeType t) {
        const bool qok = quote_ok(s);
        Floors f = floors(s);
        for (unsigned side = 0; side < 2; ++side) {
            const uint32_t act = (static_cast<uint32_t>(s.active) >> (side * 8)) & 0xFFu;
            if (!act) continue;
            Block& b = blocks_[2u * sym + side];
            const bool counts = !p_.passive_side_only || rest == side;
            prefetch_quads(b, act, counts);
            const uint32_t fb = counts ? block_d<kWide>(b, act, side ? -px : px, ts) : 0u;
            const uint32_t ca = act & ~fb;
            const uint32_t fa = (ca && qok && far_stable(s, side, t))
                              ? block_c<kWide>(b, ca, far_of(s, side), f, s) : 0u;
            if (fb) [[unlikely]] mark_d(s, sym, side, fb);
            if (fa) [[unlikely]] mark_c(s, sym, side, fa, t, threshold_of(f, s));
        }
    }

    // Fixed-address prefetches of the quad lines a decision on this side can
    // touch, issued under the (predicted) active-side branch so they leave
    // with the Sym line 0 miss instead of after it: the scalar kernel's lane
    // addresses depend on that line, and the id/generation read on a fire
    // depends on the kernel's result. Measured on a cold single-order
    // decision: three serialized memory round trips -> one (-35%).
    // always_inline for the reason given at prefetch(): out of line, the
    // call would be deleted.
    [[gnu::always_inline]] static void prefetch_quads(const Block& b, uint32_t act, bool count_d) noexcept {
        for (unsigned qi = 0; qi < 2; ++qi) {
            if (qi && !(act >> 4)) break;
            const Quad& q = b.q[qi];
            __builtin_prefetch(q.px, 0, 3);                       // px | move_min
            if (count_d) __builtin_prefetch(q.t_place, 1, 3);     // t_place | n_aggr | qh
            __builtin_prefetch(q.cold, 0, 3);                     // id / generation
        }
    }

    // The floors helpers are inlined by force: a Floors whose address is
    // never taken stays in registers, and its function needs no canary.
    [[gnu::always_inline]] static Floors floors(const Sym& s) noexcept {
        Floors f;
        f.cheap = std::max<PriceType>(s.floor0, quote_ok(s) ? s.nbo - s.nbb : PriceType{0});
        return f;
    }

    // The vol floor's inputs: at most once per update, and only after some
    // order cleared the cheap floors.
    [[gnu::always_inline]] void ensure_full(Floors& f, const Sym& s) const noexcept {
        if (f.full) return;
        if (s.var_den > 0.0) {
            f.den    = s.var_den;
            f.numhor = s.var_num * std::min(cap_ns_, p_.gap_mult * s.gap_ewma_ns);
        }
        f.full = true;
    }

    [[gnu::always_inline]] static PriceType full_threshold(const Floors& f) noexcept {      // needs f.full
        PriceType thr = f.cheap;
        if (f.numhor >= 0.0) thr = std::max(thr, static_cast<PriceType>(std::sqrt(f.numhor / f.den)));
        return thr;
    }
    // The threshold a (c) decision fired against, for the hysteresis.
    [[gnu::always_inline]] PriceType threshold_of(Floors& f, const Sym& s) const noexcept {
        ensure_full(f, s);
        return full_threshold(f);
    }

    // (a), (b): on, and their conditions; (b) also needs ours to lead its queue (abandoned()).
    bool queue_on()   const noexcept { return p_.queue_n != 0; }
    bool abandon_on() const noexcept { return p_.abandon_pct >= 0.0; }
    bool queue_full(uint32_t n, uint64_t sz, uint32_t own) const noexcept {
        return n >= p_.queue_n && static_cast<double>(sz) >= p_.queue_k * static_cast<double>(own);
    }
    bool ratio_met(uint32_t nc, uint32_t n0) const noexcept {     // false with (b) off: X < 0 would pass
        return nc != 0 && n0 != 0 && abandon_on() &&              // (no cancellation yet: no arithmetic)
               100.0 * static_cast<double>(nc) > p_.abandon_pct * static_cast<double>(n0);
    }
    // (b) holds for lane l now: the ratio, and nothing ahead of it that is
    // ours or holds a round lot.
    bool abandoned(const Sym& s, const Block& b, unsigned l, uint32_t qh) const noexcept {
        return ratio_met(b.ab.nc[l], b.ab.n0[l]) && in_book(qh) && QBlockList::clear_ahead(*queues_, qh, s.lot);
    }

    // (d)'s queue gate, read from the order's queue in the book. Off, or
    // position unknown (not in the book's queues): passes — fail toward cancel.
    // gate_d is out of line: it reads the queue with 256-bit code, which must
    // stay out of the narrow passes.
    bool aggr_gate_on() const noexcept { return p_.aggr_gate_n != 0 || p_.aggr_gate_sz != 0; }
    bool aggr_gate(uint32_t qh) const noexcept {
        if (!aggr_gate_on() || !in_book(qh)) return true;
        const QAhead a = QBlockList::ahead_of(*queues_, qh);
        return a.n >= p_.aggr_gate_n && a.sz >= p_.aggr_gate_sz;
    }
    [[gnu::noinline]] uint32_t gate_d(const Block& b, uint32_t fire) const noexcept {
        for (uint32_t m = fire; m; m &= m - 1) {
            const unsigned l = static_cast<unsigned>(std::countr_zero(m));
            if (!aggr_gate(b.q[l >> 2].qh[l & 3u])) fire &= ~(1u << l);
        }
        return fire;
    }

    // At or below kScalarLanes live orders the per-lane scalar kernel is used
    // even in the SIMD build (see the header: shorter fire-path latency).
    static constexpr int kScalarLanes = 2;

    // Which pass: wide if some live order sits beyond the first two lanes of
    // its side (bit = side * 8 + lane). Only speed depends on it.
    static bool wide(uint16_t active) noexcept { return kSimd && (active & 0xFCFCu) != 0; }

    template <bool kWide>
    [[gnu::always_inline]] uint32_t block_c(const Block& b, uint32_t cand, PriceType far, Floors& f,
                                            const Sym& s) const noexcept {
        if constexpr (kSimd && kWide) {
            if (std::popcount(cand) > kScalarLanes) return block_c_simd(b, cand, far, f, s);
        }
        return block_c_ref(b, cand, far, f, s);
    }
    template <bool kWide>
    [[gnu::always_inline]] uint32_t block_d(Block& b, uint32_t cand, PriceType trade, FeedTimeType ts) const noexcept {
        uint32_t fire;
        if constexpr (kSimd && kWide) {
            fire = std::popcount(cand) > kScalarLanes ? block_d_simd(b, cand, trade, ts)
                                                      : block_d_ref(b, cand, trade, ts);
        } else {
            fire = block_d_ref(b, cand, trade, ts);
        }
        if (fire && aggr_gate_on()) [[unlikely]] fire = gate_d(b, fire);
        return fire;
    }

    // ---- scalar kernels: live lanes only (the reference build; in the SIMD
    //      build the narrow passes, a side with <= kScalarLanes live orders
    //      in the wide ones, and evaluate()) --------------------------------------

    [[gnu::always_inline]] uint32_t block_c_ref(const Block& b, uint32_t cand, PriceType far, Floors& f,
                                                const Sym& s) const noexcept {
        uint32_t m = 0;
        for (uint32_t c = cand; c; c &= c - 1) {
            const unsigned  l = static_cast<unsigned>(std::countr_zero(c));
            const Quad&     q = b.q[l >> 2];
            const PriceType d = far - q.px[l & 3u];
            if (d > f.cheap && d > q.move_min[l & 3u]) m |= 1u << l;
        }
        if (!m) return 0;
        ensure_full(f, s);
        uint32_t out = 0;
        for (uint32_t c = m; c; c &= c - 1) {
            const unsigned l = static_cast<unsigned>(std::countr_zero(c));
            const double   x = static_cast<double>(far - b.q[l >> 2].px[l & 3u]);
            if ((x * x) * f.den > f.numhor) out |= 1u << l;
        }
        return out;
    }

    [[gnu::always_inline]] uint32_t block_d_ref(Block& b, uint32_t cand, PriceType trade,
                                                FeedTimeType ts) const noexcept {
        uint32_t fire = 0;
        for (uint32_t c = cand; c; c &= c - 1) {
            const unsigned l = static_cast<unsigned>(std::countr_zero(c)), j = l & 3u;
            Quad& q = b.q[l >> 2];
            if (!(trade > q.px[j] && ts > q.t_place[j])) continue;
            if (++q.n_aggr[j] >= p_.aggr_prints_n) fire |= 1u << l;
        }
        return fire;
    }

    // ---- AVX-512VL kernels (one ymm per 64-bit field per quad), inlined into
    //      the wide passes and reachable from nowhere else ------------------------
#if defined(__AVX512VL__) && defined(__AVX512DQ__)
    static __m256i ld(const void* p) noexcept {
        return _mm256_load_si256(static_cast<const __m256i*>(p));
    }

    [[gnu::always_inline]] uint32_t block_c_simd(const Block& b, uint32_t cand, PriceType far, Floors& f,
                                                 const Sym& s) const noexcept {
        const __m256i farv = _mm256_set1_epi64x(far), fc = _mm256_set1_epi64x(f.cheap);
        __m256i d[2] = {_mm256_setzero_si256(), _mm256_setzero_si256()};
        uint32_t m = 0;
        for (unsigned qi = 0; qi < 2; ++qi) {               // only quads with live lanes
            const __mmask8 c4 = static_cast<__mmask8>((cand >> (4 * qi)) & 0xFu);
            if (!c4) continue;
            d[qi] = _mm256_sub_epi64(farv, ld(b.q[qi].px));
            __mmask8 k = _mm256_mask_cmpgt_epi64_mask(c4, d[qi], fc);
            k = _mm256_mask_cmpgt_epi64_mask(k, d[qi], ld(b.q[qi].move_min));
            m |= static_cast<uint32_t>(k) << (4 * qi);
        }
        if (!m) return 0;
        ensure_full(f, s);
        const __m256d den = _mm256_set1_pd(f.den), rhs = _mm256_set1_pd(f.numhor);
        uint32_t out = 0;
        for (unsigned qi = 0; qi < 2; ++qi) {
            const __mmask8 c4 = static_cast<__mmask8>((m >> (4 * qi)) & 0xFu);
            if (!c4) continue;
            __m256d x = _mm256_cvtepi64_pd(d[qi]);
            x = _mm256_mul_pd(_mm256_mul_pd(x, x), den);
            out |= static_cast<uint32_t>(_mm256_mask_cmp_pd_mask(c4, x, rhs, _CMP_GT_OQ)) << (4 * qi);
        }
        return out;
    }

    [[gnu::always_inline]] uint32_t block_d_simd(Block& b, uint32_t cand, PriceType trade,
                                                 FeedTimeType ts) const noexcept {
        const __m256i tv  = _mm256_set1_epi64x(trade);
        const __m256i tsv = _mm256_set1_epi64x(static_cast<long long>(ts));
        const __m128i one = _mm_set1_epi32(1);
        const __m128i nth = _mm_set1_epi32(static_cast<int>(p_.aggr_prints_n));
        uint32_t fire = 0;
        for (unsigned qi = 0; qi < 2; ++qi) {
            const __mmask8 c4 = static_cast<__mmask8>((cand >> (4 * qi)) & 0xFu);
            if (!c4) continue;
            Quad& q = b.q[qi];
            __mmask8 k = _mm256_mask_cmpgt_epi64_mask(c4, tv, ld(q.px));      // more aggressive
            k = _mm256_mask_cmpgt_epu64_mask(k, tsv, ld(q.t_place));          // and after placement
            if (!k) continue;
            __m128i* np = reinterpret_cast<__m128i*>(q.n_aggr);
            const __m128i n = _mm_mask_add_epi32(_mm_load_si128(np), k, _mm_load_si128(np), one);
            _mm_store_si128(np, n);
            fire |= static_cast<uint32_t>(_mm_mask_cmpge_epu32_mask(k, n, nth)) << (4 * qi);
        }
        return fire;
    }
    // xmm on purpose: this is inlined into the host's handler, and a ymm value
    // there can cost the handler a realigned stack frame on every message.
    [[gnu::always_inline]] static uint32_t lanes_at_simd(const Block& b, uint32_t cand, PriceType pxs) noexcept {
        const __m128i v = _mm_set1_epi64x(pxs);
        const auto quad = [v](const Quad& q) noexcept {
            const __m128i* p = reinterpret_cast<const __m128i*>(q.px);
            return static_cast<uint32_t>(_mm_cmpeq_epi64_mask(v, _mm_load_si128(p))) |
                   (static_cast<uint32_t>(_mm_cmpeq_epi64_mask(v, _mm_load_si128(p + 1))) << 2);
        };
        uint32_t m = quad(b.q[0]);
        if (cand >> 4) m |= quad(b.q[1]) << 4;
        return m & cand;
    }
#else   // never called: kSimd is false without AVX-512VL/DQ
    static uint32_t lanes_at_simd(const Block&, uint32_t, PriceType) noexcept { return 0; }
    uint32_t block_c_simd(const Block& b, uint32_t c, PriceType fr, Floors& f,
                          const Sym& s) const noexcept { return block_c_ref(b, c, fr, f, s); }
    uint32_t block_d_simd(Block& b, uint32_t c, PriceType tr, FeedTimeType ts) const noexcept {
        return block_d_ref(b, c, tr, ts);
    }
#endif

    // ---- estimators ----------------------------------------------------------------

    // Estimator work of one update, at that update's clock t. With
    // VolSource::Venue the round-lot quote is read here, after the decision
    // pass; a side with no round lot skips the sample.
    void bookkeep(Sym& s, uint16_t sym, uint8_t bk, FeedTimeType t) const noexcept {
        if (bk & kBkQuote) {
            if constexpr (kVol == VolSource::Nbbo) {
                if (quote_ok(s)) update_mid_var(s, 0.5 * static_cast<double>(s.nbb + s.nbo), t);
            } else {
                const Nbbo v = read_venue_bbo(sym);
                if (quote_ok(v.bid, v.ask)) update_mid_var(s, 0.5 * static_cast<double>(v.bid + v.ask), t);
            }
        }
        if (bk & kBkExec) update_tau(s, t);
    }

    // Work deferred by an update whose callback is now feeding another one:
    // the symbol clock and NBBO in Sym are still that update's (the venue
    // quote is read now: see on_book_update).
    void settle(Sym& s, uint16_t sym) const noexcept {
        const uint8_t bk = s.bk;
        s.bk = 0;
        bookkeep(s, sym, bk, s.t_now);
    }

    void update_mid_var(Sym& s, double mid, FeedTimeType t) const noexcept {
        if (!s.mid_init) { s.mid_prev = mid; s.t_mid = t; s.mid_init = true; return; }
        if (mid == s.mid_prev) return;                             // one side moved, mid did not
        if (t == s.t_mid) {                                        // same instant as the last change:
            if (s.burst_open) {                                    //   revise it to the net move
                const double dm = mid - s.mid_anchor;
                s.var_num += dm * dm - s.last_inc;
                s.last_inc = dm * dm;
            }
            s.mid_prev = mid;
            return;
        }
        const double d  = ct_detail::decay(t - s.t_mid, inv_hl_vol_);
        const double dm = mid - s.mid_prev;
        s.var_num    = d * s.var_num + dm * dm;
        s.var_den    = d * s.var_den + static_cast<double>(t - s.t_mid);
        s.mid_anchor = s.mid_prev; s.last_inc = dm * dm; s.burst_open = true;
        s.mid_prev   = mid; s.t_mid = t;
    }

    void update_tau(Sym& s, FeedTimeType t) const noexcept {
        if (!s.have_exec) { s.have_exec = true; s.t_event = t; return; }
        const FeedTimeType gap = t - s.t_event;                    // symbol clock: never negative
        if (gap <= p_.trade_coalesce_ns) return;                   // same trade event
        const double g = static_cast<double>(gap);
        s.gap_ewma_ns = s.gap_seeded ? s.gap_ewma_ns + p_.gap_alpha * (g - s.gap_ewma_ns) : g;
        s.gap_seeded  = true;
        s.t_event     = t;
    }

    // ---- decisions -------------------------------------------------------------------

    void mark_a(Sym& s, uint16_t sym, unsigned side, uint32_t m) noexcept {
        const uint16_t bits = static_cast<uint16_t>(m << (side * 8));
        s.active = static_cast<uint16_t>(s.active & ~bits);
        s.barm   = static_cast<uint16_t>(s.barm & ~bits);
        s.pend_a = static_cast<uint16_t>(s.pend_a | bits);
        if (!s.active) retrack(sym);
    }
    void mark_b(Sym& s, uint16_t sym, unsigned side, uint32_t m) noexcept {
        const uint16_t bits = static_cast<uint16_t>(m << (side * 8));
        s.active = static_cast<uint16_t>(s.active & ~bits);
        s.barm   = static_cast<uint16_t>(s.barm & ~bits);
        s.pend_b = static_cast<uint16_t>(s.pend_b | bits);
        if (!s.active) retrack(sym);
    }
    void mark_c(Sym& s, uint16_t sym, unsigned side, uint32_t m, FeedTimeType t, PriceType thr) noexcept {
        const uint16_t bits = static_cast<uint16_t>(m << (side * 8));
        s.active = static_cast<uint16_t>(s.active & ~bits);
        s.barm   = static_cast<uint16_t>(s.barm & ~bits);
        s.pend_c = static_cast<uint16_t>(s.pend_c | bits);
        s.last_cancel_thr[side] = thr;
        s.t_last_cancel[side]   = t;
        if (!s.active) retrack(sym);
    }
    void mark_d(Sym& s, uint16_t sym, unsigned side, uint32_t m) noexcept {
        const uint16_t bits = static_cast<uint16_t>(m << (side * 8));
        s.active = static_cast<uint16_t>(s.active & ~bits);
        s.barm   = static_cast<uint16_t>(s.barm & ~bits);
        s.pend_d = static_cast<uint16_t>(s.pend_d | bits);
        if (!s.active) retrack(sym);
    }

    // Deliver pending decisions. Re-reads the pending bits after every
    // callback: a callback that frees an order clears its bit, and a nested
    // update that decides more sets bits that either loop may deliver — each
    // exactly once.
    template <class G>
    [[gnu::always_inline]]
    void deliver(uint16_t sym, G& cb) {
        Sym& s = syms_[sym];
        for (;;) {
            const uint32_t pa = s.pend_a, pb = s.pend_b, pc = s.pend_c, pd = s.pend_d;
            const uint32_t m = pa | pb | pc | pd;
            if (!m) return;
            const unsigned i   = static_cast<unsigned>(std::countr_zero(m));
            const uint32_t bit = 1u << i;
            s.pend_a = static_cast<uint16_t>(pa & ~bit);
            s.pend_b = static_cast<uint16_t>(pb & ~bit);
            s.pend_c = static_cast<uint16_t>(pc & ~bit);
            s.pend_d = static_cast<uint16_t>(pd & ~bit);
            const unsigned side = i >> 3, lane = i & 7u;
            const Cold& c = blocks_[2u * sym + side].q[lane >> 2].cold[lane & 3u];
            cb(make_handle(c.gen, slot_of(sym, side, lane)), c.id,
               (pa & bit) ? Decision::CancelQueue : (pb & bit) ? Decision::CancelAbandoned
                          : (pd & bit) ? Decision::CancelOutcompeted : Decision::CancelDepth);
        }
    }

    CancelParams          p_;
    double                inv_hl_vol_ = 0.0, cap_ns_ = 0.0;
    std::vector<Sym>      syms_;
    std::vector<Block>    blocks_;
    std::vector<uint8_t>  sides_;              // per symbol: bit s = side s tracks orders,
                                               //   bit 2+s = its lanes 4-7 in use; for prefetch()
    std::vector<uint64_t> track_;              // per symbol: tracking() bit
    QBlockPool*           queues_;
    std::size_t           max_live_, live_ = 0, place_failures_ = 0;
    std::size_t           late_placed_ = 0;
};

using CancelTactic = CancelTacticT<kCancelTacticSimd, kCancelVolSource, kSizeUpKeepsPriority>;

// ---------------------------------------------------------------------------
// [5] JoinTactic — join signals on improved price levels
//
// Signal: one order of at least the symbol's round lot lands at a price
// better than its side's best on this venue — a higher bid, a lower offer —
// so a new improved level forms with a full round lot from a single order.
// An Add qualifies, and so does the new order of a Replace. It must improve
// on an existing best: an order into an empty side signals nothing.
//
// Lifetime: a signal lives until the join tactic reads its kill flag
// (SymbolState::tac.side[s].kill — the strategy's; kill() sets it for now),
// which it does on the side's next execution, on its next improving Add (of
// an armed symbol) and in poll(). While it lives, new improved levels on its
// side are not new signals; the next improvement after the kill is.
// Disarming a symbol stops new signals only: a live signal keeps counting
// until killed.
//
// The flag is a plain byte: it says "end this side's signal", not which
// one. Set from the feed thread (a callback) that is exact. Set from another
// thread it can land after the signal it was meant for has ended and the
// next has begun, and end that one; if the strategy runs there, have the
// real flag carry the signal's JoinSide::seq and compare it in reap(). The
// same applies to reading JoinSide from another thread: take what is needed
// in on_signal, or the feed thread pays a line transfer per execution.
//
// Our own orders: the tactic cannot tell them from the market's. An
// improving round-lot order of ours passed to on_add signals like any other;
// skip the call for it unless that is wanted.
//
// At the signal the tactic stores in the symbol's SymbolState::tac, for that
// side, the signal price and the improving order's size, the time, and this
// venue's round-lot far touch (the opposite side's round-lot best: its
// price, and the size from that side's top through it) read with
// getVenueRoundLot, then calls the host. From then on it counts the
// executions at the signal price and beyond it (more aggressive: higher for
// a buy signal, lower for a sell), the latter split by which side's resting
// order traded, and keeps the 16 most aggressive execution prices by resting
// side (JoinTape). From those the cancel tactic starts (d) for an order of
// ours placed while the signal lives, at any price: the executions more
// aggressive than that price since the signal (CancelParams::aggr_from_signal).
//
// Host calls, per message, after applying it to the book (any symbol: the
// tactic's bitmaps make one it does not watch a bit test):
//   Add, and the new order of a Replace:
//       on_add(sym, side, px, sz, prev_best, ts, on_signal)
//       prev_best: that side's best on this venue before this order (for a
//       Replace, after the old order left); 0 if the side was empty.
//       on_signal: void(uint16_t sym, Side, const JoinSide&).
//   Execute: on_execute(sym, resting_side, px, sz)
//       px: the executed order's price; sz: the shares executed.
// An Add costs a bit test and a compare with prev_best; only one that
// improves on it reads the side's line, and only a signal reads the far
// touch. An execution of a symbol without a live signal costs a bit test;
// with one, a line and a tape insert per live side. Those lines live in the
// symbol's SymbolState and are usually not in cache when a quiet symbol
// trades: prefetch(sym), called when the message's symbol is decoded, warms
// them (a bit test for a symbol without a live signal).
// ---------------------------------------------------------------------------

// One side's round-lot best on this venue: its price (0 = none) and the size
// from the side's top through that price.
struct RlQuote { PriceType px; uint64_t sz; };

// getVenueRoundLot(symst_idx, sd): host-defined, used only on a join signal.
// With this file's MarketDataSystem `mds`:
//   md::RlQuote md::getVenueRoundLot(uint16_t s, Side sd) noexcept {
//       const auto r = mds.roundlot_best(s, sd);
//       return r ? md::RlQuote{r->price, r->cum_size} : md::RlQuote{0, 0};
//   }
RlQuote getVenueRoundLot(uint16_t symst_idx, Side sd) noexcept;

class JoinTactic {
public:
    explicit JoinTactic(std::size_t nsymbols)
        : armed_((nsymbols + 63) / 64, 0), live_((2 * nsymbols + 63) / 64, 0) {
        assert(nsymbols <= (std::size_t{1} << 14));
    }

    // Watch a symbol for new join signals, or stop. A live signal is left
    // as it is: it ends only on its kill flag. Arming copies the round lot
    // from SymbolState for the next signals (re-arm after changing it).
    void arm(uint16_t sym, bool on) noexcept {
        uint64_t& w = armed_[sym >> 6];
        const uint64_t bit = 1ull << (sym & 63u);
        if (!on) { w &= ~bit; return; }
        SymbolState& st = getSymbolState(sym);
        for (JoinSide& js : st.tac.side) js.lot = st.round_lot;
        w |= bit;
    }
    [[nodiscard]] bool armed(uint16_t sym) const noexcept {
        return (armed_[sym >> 6] >> (sym & 63u)) & 1u;
    }
    [[nodiscard]] bool live(uint16_t sym, Side sd) const noexcept {
        const unsigned k = 2u * sym + static_cast<unsigned>(sd);
        return (live_[k >> 6] >> (k & 63u)) & 1u;
    }

    // on_add and on_execute are inlined at every call site, as
    // CancelTactic::on_book_update is: a symbol the tactic does not watch then
    // costs the bit test and no call, however many handlers call them. What
    // lies behind the tests — a signal's start, the counting — is out of line.
    template <class F>
    [[gnu::always_inline]]
    void on_add(uint16_t sym, Side sd, PriceType px, uint32_t sz, PriceType prev_best,
                FeedTimeType ts, F&& on_signal) {
        if (!armed(sym)) return;
        if (sd == Side::Buy ? px <= prev_best : px >= prev_best) return;   // most Adds: no line read
        if (prev_best == 0) return;                // empty side (an empty offer side failed above)
        const unsigned side = static_cast<unsigned>(sd);
        TacticShared& t = getSymbolState(sym).tac;
        JoinSide& js = t.side[side];
        if (live(sym, sd) && !reap(sym, side, js)) return;   // a live signal: not a new one
        if (sz < js.lot) return;
        start(sym, sd, t, px, sz, ts);
        on_signal(sym, sd, static_cast<const JoinSide&>(js));
    }

    [[gnu::always_inline]]
    void on_execute(uint16_t sym, Side rest, PriceType px, uint32_t sz) noexcept {
        const unsigned k = 2u * sym;
        const uint32_t lv = static_cast<uint32_t>(live_[k >> 6] >> (k & 63u)) & 3u;   // both sides
        if (!lv) return;
        TacticShared& t = getSymbolState(sym).tac;
        if (lv & 1u) count<true>(sym, t, px, sz, rest == Side::Buy);
        if (lv & 2u) count<false>(sym, t, px, sz, rest == Side::Sell);
    }

    // Decode stage: warm what an execution of this symbol will touch — the
    // signal line and the tape of each side with a live signal — ahead of
    // on_execute. A symbol without a live signal costs the bit test. Inlined
    // by force, for the reason given at CancelTactic::prefetch.
    [[gnu::always_inline]] void prefetch(uint16_t sym) const noexcept {
        const unsigned k = 2u * sym;
        const uint32_t lv = static_cast<uint32_t>(live_[k >> 6] >> (k & 63u)) & 3u;
        if (!lv) [[likely]] return;
        const TacticShared& t = getSymbolState(sym).tac;
        for (unsigned side = 0; side < 2; ++side) {
            if (!((lv >> side) & 1u)) continue;
            __builtin_prefetch(&t.side[side], 1, 3);
            __builtin_prefetch(t.tape[side].same, 1, 3);
            __builtin_prefetch(t.tape[side].opp, 1, 3);
        }
    }

    // Read the symbol's kill flags now (on_add and on_execute also do).
    void poll(uint16_t sym) noexcept {
        TacticShared& t = getSymbolState(sym).tac;
        for (unsigned side = 0; side < 2; ++side)
            if (live(sym, static_cast<Side>(side))) reap(sym, side, t.side[side]);
    }

    // The kill flag of a side's signal belongs to the strategy, which sets it
    // (a relaxed atomic store) when the signal should end; this does it for
    // now. The signal ends when the join tactic next reads it; the cancel
    // tactic stops starting (d) from it at once.
    static void kill(uint16_t sym, Side sd) noexcept {
        __atomic_store_n(&getSymbolState(sym).tac.side[static_cast<unsigned>(sd)].kill, uint8_t{1},
                         __ATOMIC_RELAXED);
    }
    [[nodiscard]] const JoinSide& signal(uint16_t sym, Side sd) const noexcept {
        return getSymbolState(sym).tac.side[static_cast<unsigned>(sd)];
    }
    [[nodiscard]] const JoinTape& tape(uint16_t sym, Side sd) const noexcept {
        return getSymbolState(sym).tac.tape[static_cast<unsigned>(sd)];
    }
    // What the cancel tactic starts (d) at for an order at px on side sd.
    [[nodiscard, gnu::noinline]] uint32_t aggr_since(uint16_t sym, Side sd, PriceType px,
                                                     bool same_only) const noexcept {
        return join_aggr(getSymbolState(sym).tac, static_cast<unsigned>(sd), px, same_only);
    }

private:
    void set_live(uint16_t sym, unsigned side, bool on) noexcept {
        const unsigned k = 2u * sym + side;
        uint64_t& w = live_[k >> 6];
        w = on ? (w | (1ull << (k & 63u))) : (w & ~(1ull << (k & 63u)));
    }
    // A signal begins. Out of line, as count() is: they hold the tape's
    // 256-bit code, which inlined into the host's handler can cost it a
    // realigned stack frame on every message.
    [[gnu::noinline]] void start(uint16_t sym, Side sd, TacticShared& t, PriceType px, uint32_t sz,
                                 FeedTimeType ts) noexcept {
        const unsigned side = static_cast<unsigned>(sd);
        JoinSide& js = t.side[side];
        const RlQuote f = getVenueRoundLot(sym, sd == Side::Buy ? Side::Sell : Side::Buy);
        js.px = px; js.far_px = f.px; js.t = ts; js.sz = sz;
        js.far_sz = static_cast<uint32_t>(std::min<uint64_t>(f.sz, 0xFFFFFFFFu));
        __atomic_store_n(&js.kill, uint8_t{0}, __ATOMIC_RELAXED);   // a flag set with no live signal is stale
        ++js.seq;
        js.n_at = js.n_thru_same = js.n_thru_opp = 0;
        js.sh_at = js.sh_thru_same = js.sh_thru_opp = 0;
        TacticShared::clear_tape(t.tape[side]);
        set_live(sym, side, true);
    }
    // A live side's kill flag: if set, end the signal and clear the flag.
    bool reap(uint16_t sym, unsigned side, JoinSide& js) noexcept {
        if (!__atomic_load_n(&js.kill, __ATOMIC_RELAXED)) return false;
        js.px = 0;
        __atomic_store_n(&js.kill, uint8_t{0}, __ATOMIC_RELAXED);
        set_live(sym, side, false);
        return true;
    }
    // kBuy: the buy side's signal (beyond = higher). same: the resting order
    // that traded is on the signal's side.
    template <bool kBuy>
    [[gnu::noinline]] void count(uint16_t sym, TacticShared& t, PriceType px, uint32_t sz, bool same) noexcept {
        constexpr unsigned side = kBuy ? 0u : 1u;
        JoinSide& js = t.side[side];
        if (reap(sym, side, js)) return;
        tape_insert(same ? t.tape[side].same : t.tape[side].opp, tape_off(px, js.px, !kBuy));
        if (px == js.px) { ++js.n_at; js.sh_at += sz; return; }
        if (kBuy ? px < js.px : px > js.px) return;              // short of the signal price
        if (same) { ++js.n_thru_same; js.sh_thru_same += sz; }
        else      { ++js.n_thru_opp;  js.sh_thru_opp  += sz; }
    }

    std::vector<uint64_t> armed_;
    std::vector<uint64_t> live_;               // bit 2*sym + side: a live signal
};

}  // namespace md
