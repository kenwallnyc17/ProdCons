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
//      round lot, market state) plus bid and ask books.
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
// Types follow the user's definitions exactly; PrcLevelGeneric drops the {}
// default member initializers so the type stays trivially default
// constructible (zeroed storage == valid empty state; required for static /
// hugepage placement, per earlier findings).
//
// Single writer. Build: -std=c++23 -O3 -march=native (AVX2; scalar fallback).

#pragma once

#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

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

struct SymbolState {
    LadderBook books[2];           // [Buy], [Sell]
    PriceType  tick        = 100;
    uint32_t   round_lot   = 100;
    uint8_t    market_state = 0;   // trading/halted/... user-defined
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

}  // namespace md
