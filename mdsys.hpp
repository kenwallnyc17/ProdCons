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
        std::memset(&p, 0, sizeof(Page));
        ++used_;
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

class LadderBook {
    static constexpr uint32_t kOvfBit = 0x80000000u;
public:
    // window: tick count covered by the ladder; pow2, multiple of 4096
    // (64*64 for the two-level bitmap), anchored at first price seen.
    void configure(bool is_bid, PriceType tick, uint32_t window) {
        assert(std::has_single_bit(window) && window >= 4096 && window <= (1u << 20));
        is_bid_ = is_bid;
        tick_   = tick;
        window_ = window;
        dir_.assign(window / PagePool::kSlots, 0);
        occ_.assign(window / 64, 0);
        sum_.assign(window / 4096, 0);
        best_ = -1;
        anchor_ = -1;
        ovf_.clear();
        ovf_live_ = 0;
    }

    // Add flow at price; creates the level if absent. Returns ordq handle.
    uint32_t add(PagePool& pp, PriceType prc, SizeType sz,
                 FeedTimeType ft, UniqueUpdateID uu) noexcept {
        if (anchor_ < 0) {
            const PriceType half = static_cast<PriceType>(window_ / 2) * tick_;
            anchor_ = prc > half ? prc - half : 0;
        }
        const PriceType rel = prc - anchor_;
        if (rel < 0 || rel % tick_ != 0 ||
            rel / tick_ >= static_cast<PriceType>(window_)) [[unlikely]]
            return ovf_add(prc, sz, ft, uu);               // fail-closed path

        const uint32_t idx  = static_cast<uint32_t>(rel / tick_);   // Add-only division
        const uint32_t di   = idx >> 9;
        const uint32_t slot = idx & (PagePool::kSlots - 1);
        if (dir_[di] == 0) {
            const uint32_t pi = pp.alloc();
            assert(pi != 0 && "page pool exhausted — size npages higher");
            dir_[di] = pi;
        }
        PrcLevelGeneric& L = pp.at(dir_[di]).s[slot];
        if (!bit_test(idx)) {                              // new level
            bit_set(idx);
            ++pp.at(dir_[di]).live;
            L.agg_sz = sz; L.num_orders = 1; L.plist = nullptr;
            if (best_ < 0 || (is_bid_ ? static_cast<int32_t>(idx) > best_
                                      : static_cast<int32_t>(idx) < best_))
                best_ = static_cast<int32_t>(idx);
        } else {
            L.agg_sz += sz;
            assert(L.num_orders < static_cast<NumOrdersType>(~NumOrdersType{0}));
            L.num_orders += 1;
        }
        L.feed_time = ft; L.uuid = uu;
        return idx;
    }

    // Reduce through the handle (E/C/X/D path — no price math, no division).
    void reduce(PagePool& pp, uint32_t ordq, SizeType sz, bool order_gone,
                FeedTimeType ft, UniqueUpdateID uu) noexcept {
        if (ordq & kOvfBit) [[unlikely]] { ovf_reduce(ordq & ~kOvfBit, sz, order_gone, ft, uu); return; }
        const uint32_t idx  = ordq;
        const uint32_t di   = idx >> 9;
        const uint32_t slot = idx & (PagePool::kSlots - 1);
        assert(dir_[di] != 0 && bit_test(idx));
        PrcLevelGeneric& L = pp.at(dir_[di]).s[slot];
        assert(L.agg_sz >= sz);
        L.agg_sz -= sz;
        L.feed_time = ft; L.uuid = uu;
        if (order_gone) {
            assert(L.num_orders > 0);
            if (--L.num_orders == 0) {                     // level death
                assert(L.agg_sz == 0);
                bit_clear(idx);
                PagePool::Page& pg = pp.at(dir_[di]);
                if (--pg.live == 0) { pp.free(dir_[di]); dir_[di] = 0; }  // recycle
                if (static_cast<int32_t>(idx) == best_)    // inside recovery
                    best_ = is_bid_ ? next_below(idx) : next_above(idx);
            }
        }
    }

    // Tick-processor stage: address of the level a handle points at, computed
    // WITHOUT touching the level line — so it can be prefetched ahead of the
    // mutation. nullptr for overflow levels / unbound pages (skip prefetch).
    [[nodiscard]] const PrcLevelGeneric*
    level_addr(const PagePool& pp, uint32_t ordq) const noexcept {
        if (ordq & kOvfBit) [[unlikely]] return nullptr;
        const uint32_t di = ordq >> 9;
        if (dir_[di] == 0) return nullptr;
        return &pp.at(dir_[di]).s[ordq & (PagePool::kSlots - 1)];
    }
    // Same, from a price (Add path): resolvable once the anchor exists and
    // the target page is bound; a miss just skips the prefetch.
    [[nodiscard]] const PrcLevelGeneric*
    level_addr_for_price(const PagePool& pp, PriceType prc) const noexcept {
        if (anchor_ < 0) return nullptr;
        const PriceType rel = prc - anchor_;
        if (rel < 0 || rel % tick_ != 0 ||
            rel / tick_ >= static_cast<PriceType>(window_)) return nullptr;
        const uint32_t idx = static_cast<uint32_t>(rel / tick_);
        if (dir_[idx >> 9] == 0) return nullptr;
        return &pp.at(dir_[idx >> 9]).s[idx & (PagePool::kSlots - 1)];
    }

    // Best price on this side (ladder best merged with overflow).
    [[nodiscard]] std::optional<PriceType> best() const noexcept {
        std::optional<PriceType> r;
        if (best_ >= 0) r = anchor_ + static_cast<PriceType>(best_) * tick_;
        if (ovf_live_ != 0) [[unlikely]]
            for (const auto& e : ovf_)
                if (e.prc != 0 && (!r || (is_bid_ ? e.prc > *r : e.prc < *r)))
                    r = e.prc;
        return r;
    }

    // Visit levels best->outward: f(prc, const PrcLevelGeneric&) -> bool.
    template <class F>
    void for_each_from_best(const PagePool& pp, F&& f) const {
        // rare-path merge with overflow: collect live overflow, insertion-sort
        std::vector<PriceType> ov;
        if (ovf_live_ != 0) [[unlikely]] {
            for (const auto& e : ovf_) if (e.prc != 0) ov.push_back(e.prc);
            for (std::size_t i = 1; i < ov.size(); ++i)     // tiny n
                for (std::size_t j = i; j > 0 && (is_bid_ ? ov[j] > ov[j-1]
                                                          : ov[j] < ov[j-1]); --j)
                    std::swap(ov[j], ov[j-1]);
        }
        std::size_t oi = 0;
        int32_t cur = best_;
        while (cur >= 0 || oi < ov.size()) {
            const PriceType lp = cur >= 0
                ? anchor_ + static_cast<PriceType>(cur) * tick_ : 0;
            const bool take_ovf =
                oi < ov.size() && (cur < 0 || (is_bid_ ? ov[oi] > lp : ov[oi] < lp));
            if (take_ovf) {
                const auto* e = ovf_find(ov[oi]);
                if (!f(ov[oi], e->lvl)) return;
                ++oi;
            } else {
                const uint32_t di = static_cast<uint32_t>(cur) >> 9;
                if (!f(lp, pp.at(dir_[di]).s[static_cast<uint32_t>(cur)
                                             & (PagePool::kSlots - 1)])) return;
                cur = is_bid_ ? next_below(static_cast<uint32_t>(cur))
                              : next_above(static_cast<uint32_t>(cur));
            }
        }
    }

    struct RlBest { PriceType price; uint64_t cum_size; uint32_t levels_used; };
    [[nodiscard]] std::optional<RlBest>
    roundlot_best(const PagePool& pp, uint32_t lot) const {
        uint64_t cum = 0; uint32_t used = 0; std::optional<RlBest> out;
        for_each_from_best(pp, [&](PriceType p, const PrcLevelGeneric& L) {
            cum += L.agg_sz; ++used;
            if (cum >= lot) { out = RlBest{p, cum, used}; return false; }
            return true;
        });
        return out;
    }

    // Test/introspection: level at an exact price (ladder or overflow).
    [[nodiscard]] const PrcLevelGeneric*
    level_at(const PagePool& pp, PriceType prc) const noexcept {
        if (anchor_ >= 0) {
            const PriceType rel = prc - anchor_;
            if (rel >= 0 && rel % tick_ == 0 &&
                rel / tick_ < static_cast<PriceType>(window_)) {
                const uint32_t idx = static_cast<uint32_t>(rel / tick_);
                if (bit_test(idx) && dir_[idx >> 9])
                    return &pp.at(dir_[idx >> 9]).s[idx & (PagePool::kSlots - 1)];
                return nullptr;
            }
        }
        const auto* e = ovf_find(prc);
        return e ? &e->lvl : nullptr;
    }

    [[nodiscard]] std::size_t overflow_live() const noexcept { return ovf_live_; }

private:
    // ---- bitmaps ---------------------------------------------------------
    [[nodiscard]] bool bit_test(uint32_t i) const noexcept {
        return (occ_[i >> 6] >> (i & 63)) & 1u;
    }
    void bit_set(uint32_t i) noexcept {
        occ_[i >> 6] |= 1ull << (i & 63);
        sum_[i >> 12] |= 1ull << ((i >> 6) & 63);
    }
    void bit_clear(uint32_t i) noexcept {
        occ_[i >> 6] &= ~(1ull << (i & 63));
        if (occ_[i >> 6] == 0) sum_[i >> 12] &= ~(1ull << ((i >> 6) & 63));
    }
    // next populated tick strictly below/above `from`; -1 if none.
    [[nodiscard]] int32_t next_below(uint32_t from) const noexcept {
        int32_t w = static_cast<int32_t>(from >> 6);
        const uint32_t b = from & 63;
        uint64_t m = b ? (occ_[w] & ((1ull << b) - 1)) : 0;
        if (m) return (w << 6) + 63 - std::countl_zero(m);
        int32_t sw = w >> 6;
        uint64_t sm = (w & 63) ? (sum_[sw] & ((1ull << (w & 63)) - 1)) : 0;
        for (;;) {
            if (sm) {
                const int32_t w2 = (sw << 6) + 63 - std::countl_zero(sm);
                return (w2 << 6) + 63 - std::countl_zero(occ_[w2]);
            }
            if (--sw < 0) return -1;
            sm = sum_[sw];
        }
    }
    [[nodiscard]] int32_t next_above(uint32_t from) const noexcept {
        const int32_t nwords = static_cast<int32_t>(occ_.size());
        int32_t w = static_cast<int32_t>(from >> 6);
        const uint32_t b = from & 63;
        uint64_t m = (b == 63) ? 0 : (occ_[w] & ~((2ull << b) - 1));
        if (m) return (w << 6) + std::countr_zero(m);
        int32_t sw = w >> 6;
        uint64_t sm = ((w & 63) == 63) ? 0 : (sum_[sw] & ~((2ull << (w & 63)) - 1));
        const int32_t nsum = static_cast<int32_t>(sum_.size());
        for (;;) {
            if (sm) {
                const int32_t w2 = (sw << 6) + std::countr_zero(sm);
                if (w2 < nwords) return (w2 << 6) + std::countr_zero(occ_[w2]);
            }
            if (++sw >= nsum) return -1;
            sm = sum_[sw];
        }
    }

    // ---- overflow store (rare; zeros-mean-deleted, stable slots) ---------
    struct Ovf { PriceType prc; PrcLevelGeneric lvl; };

    uint32_t ovf_add(PriceType prc, SizeType sz,
                     FeedTimeType ft, UniqueUpdateID uu) noexcept {
        uint32_t slot = ~0u, freeslot = ~0u;
        for (uint32_t i = 0; i < ovf_.size(); ++i) {
            if (ovf_[i].prc == prc) { slot = i; break; }
            if (ovf_[i].prc == 0 && freeslot == ~0u) freeslot = i;
        }
        if (slot == ~0u) {
            if (freeslot != ~0u) slot = freeslot;
            else { slot = static_cast<uint32_t>(ovf_.size()); ovf_.push_back({}); }
            ovf_[slot].prc = prc;
            ovf_[slot].lvl = PrcLevelGeneric{};
            ovf_[slot].lvl.agg_sz = sz; ovf_[slot].lvl.num_orders = 1;
            ++ovf_live_;
        } else {
            ovf_[slot].lvl.agg_sz += sz;
            ovf_[slot].lvl.num_orders += 1;
        }
        ovf_[slot].lvl.feed_time = ft; ovf_[slot].lvl.uuid = uu;
        return kOvfBit | slot;
    }
    void ovf_reduce(uint32_t slot, SizeType sz, bool gone,
                    FeedTimeType ft, UniqueUpdateID uu) noexcept {
        Ovf& e = ovf_[slot];
        assert(e.prc != 0 && e.lvl.agg_sz >= sz);
        e.lvl.agg_sz -= sz;
        e.lvl.feed_time = ft; e.lvl.uuid = uu;
        if (gone && --e.lvl.num_orders == 0) { e.prc = 0; --ovf_live_; }
    }
    [[nodiscard]] const Ovf* ovf_find(PriceType prc) const noexcept {
        for (const auto& e : ovf_) if (e.prc == prc) return &e;
        return nullptr;
    }

    bool       is_bid_ = true;
    PriceType  tick_   = 100;      // 1c in ITCH 1/10000$ units
    PriceType  anchor_ = -1;       // set on first add
    uint32_t   window_ = 32768;
    int32_t    best_   = -1;       // rel tick index of ladder best
    std::vector<uint32_t> dir_;    // page-pool index + 1; 0 = unbound
    std::vector<uint64_t> occ_, sum_;
    std::vector<Ovf> ovf_;
    std::size_t ovf_live_ = 0;
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
                     std::size_t npages, std::size_t swan_buckets)
        : orders_(peak_orders, swan_buckets), pages_(npages), syms_(nsymbols) {
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
                                  .add(pages_, prc, sz, ft, uu);
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
            bk.reduce(pages_, h.b->ordq[h.lane], sz, /*order_gone*/true, ft, uu);
            return orders_.erase(ref, nullptr, nullptr, nullptr, nullptr, nullptr);
        }
        bk.reduce(pages_, h.b->ordq[h.lane], sz, /*order_gone*/false, ft, uu);
        h.b->sz[h.lane] = remain;
        return true;
    }

    bool on_delete(OrderRef ref, FeedTimeType ft, UniqueUpdateID uu) noexcept {
        PriceType prc; SizeType sz; uint32_t ordq; uint16_t sy; uint8_t sd;
        if (!orders_.erase(ref, &prc, &sz, &ordq, &sy, &sd)) return false;
        sym(sy).books[sd].reduce(pages_, ordq, sz, /*order_gone*/true, ft, uu);
        return true;
    }

    // ITCH U: delete old ref, add new ref at new prc/sz on the same side.
    bool on_replace(OrderRef old_ref, OrderRef new_ref, PriceType prc,
                    SizeType sz, FeedTimeType ft, UniqueUpdateID uu) noexcept {
        PriceType op; SizeType osz; uint32_t oq; uint16_t sy; uint8_t sd;
        if (!orders_.erase(old_ref, &op, &osz, &oq, &sy, &sd)) return false;
        sym(sy).books[sd].reduce(pages_, oq, osz, /*order_gone*/true, ft, uu);
        const uint32_t nq = sym(sy).books[sd].add(pages_, prc, sz, ft, uu);
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
                                    .level_addr_for_price(pages_, x.prc);
                if (L) __builtin_prefetch(L, 1, 3);
                return {};
            }
            auto h = orders_.find(x.ref);
            if (h) {
                const SymbolState& s = sym(h.b->symst[h.lane]);
                const LadderBook& bk = s.books[h.b->side[h.lane]];
                if (const auto* L = bk.level_addr(pages_, h.b->ordq[h.lane]))
                    __builtin_prefetch(L, 1, 3);
                if (x.type == TickMsg::T::Replace)          // new level too
                    if (const auto* L2 = bk.level_addr_for_price(pages_, x.prc))
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
        return s.books[static_cast<unsigned>(sd)].roundlot_best(pages_, s.round_lot);
    }
    template <class F>
    void for_each_level(uint16_t symst_idx, Side sd, F&& f) const {
        csym(symst_idx).books[static_cast<unsigned>(sd)]
            .for_each_from_best(pages_, static_cast<F&&>(f));
    }
    [[nodiscard]] const PrcLevelGeneric*
    level_at(uint16_t symst_idx, Side sd, PriceType prc) const noexcept {
        return csym(symst_idx).books[static_cast<unsigned>(sd)].level_at(pages_, prc);
    }

    OrderTable&       order_table()       noexcept { return orders_; }
    const OrderTable& order_table() const noexcept { return orders_; }
    const PagePool&   page_pool()   const noexcept { return pages_; }
    SymbolState&      sym(uint16_t i)        noexcept { assert(i < syms_.size()); return syms_[i]; }
    const SymbolState& csym(uint16_t i) const noexcept { assert(i < syms_.size()); return syms_[i]; }

private:
    OrderTable  orders_;
    PagePool    pages_;
    std::vector<SymbolState> syms_;
};

}  // namespace md
