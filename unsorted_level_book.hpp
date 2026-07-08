// unsorted_level_book.hpp — per-symbol, per-side price-level store with NO
// sorted invariant. Updates are O(1) array writes; price ordering is computed
// only when asked for, via SIMD masked max/min scans.
//
// Design (one instance per symbol per side):
//   * SoA: int32 prices[] + uint32 sizes[] + uint32 norders[], dense-ish array.
//   * Slots are STABLE: emptied levels become tombstones on a freelist and are
//     reused in place. (Order.lvl indices point at slots; swap-remove would
//     retarget other levels' orders.) The tombstone price doubles as the SIMD
//     mask: kEmptyBid = INT32_MIN never wins a max-scan, kEmptyAsk = INT32_MAX
//     never wins a min-scan — so empty slots cost nothing in the query path.
//   * upsert_level: SIMD equality-scan for the price; hit -> add size; miss ->
//     reuse a freelist slot or append. Returns the stable slot index for
//     Order.lvl.
//   * roundlot_best: repeated masked max-scan (bids) / min-scan (asks):
//     find best remaining price, accumulate its size, tighten the bound,
//     repeat until cum >= round_lot. k passes for k levels consumed; liquid
//     books hit a round lot in 1-3 passes.
//
// Round-lot size is a per-symbol parameter: under the amended SEC round-lot
// tiers it varies with price band (100 shares for most names, smaller for
// high-priced ones) — verify current tiers against the live rule set.
//
// AVX2 with scalar fallback. AVX-512 deliberately omitted: fused off on client
// Alder Lake, and the arrays here are small enough that 8-wide is plenty.
//
// Single writer per book. Build: -std=c++23 -O3 -march=native (or -mavx2).

#pragma once

#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace md {

enum class BookSide : std::uint8_t { Bid = 0, Ask = 1 };

struct RoundLotBest {
    std::int32_t  price;       // price at which cumulative size first >= lot
    std::uint64_t cum_size;    // cumulative size including this level
    std::uint32_t levels_used; // how many levels were consumed
};

template <BookSide Side>
class UnsortedLevelBook {
    static constexpr bool kBid = (Side == BookSide::Bid);
    // Tombstone/empty sentinel: loses every comparison for this side's scan.
    static constexpr std::int32_t kEmpty =
        kBid ? std::numeric_limits<std::int32_t>::min()
             : std::numeric_limits<std::int32_t>::max();

public:
    explicit UnsortedLevelBook(std::size_t reserve_levels = 512) {
        const std::size_t cap = round_up8(reserve_levels ? reserve_levels : 8);
        prices_.assign(cap, kEmpty);
        sizes_.assign(cap, 0);
        norders_.assign(cap, 0);
    }

    // ---- O(1)-ish updates (the whole point of dropping the sort) ---------

    // Add order flow into a level; creates the level if absent.
    // Returns the STABLE slot index to store in Order.lvl.
    std::uint32_t add(std::int32_t prc, std::uint32_t sz) noexcept {
        assert(prc != kEmpty);
        if (std::int32_t idx = find_price(prc); idx >= 0) {
            sizes_[static_cast<std::size_t>(idx)]   += sz;
            norders_[static_cast<std::size_t>(idx)] += 1;
            return static_cast<std::uint32_t>(idx);
        }
        std::uint32_t slot;
        if (!free_.empty()) { slot = free_.back(); free_.pop_back(); }
        else {
            if (used_ == prices_.size()) grow();
            slot = static_cast<std::uint32_t>(used_++);
        }
        prices_[slot]  = prc;
        sizes_[slot]   = sz;
        norders_[slot] = 1;
        ++live_;
        return slot;
    }

    // Reduce a level by slot index (cancel/execute path: Order.lvl in hand,
    // no price scan). full_out => the order is fully gone from the level.
    void reduce(std::uint32_t slot, std::uint32_t sz, bool order_gone) noexcept {
        assert(slot < used_ && prices_[slot] != kEmpty);
        assert(sizes_[slot] >= sz);
        sizes_[slot] -= sz;
        if (order_gone) {
            assert(norders_[slot] > 0);
            if (--norders_[slot] == 0) {          // level empty -> tombstone
                assert(sizes_[slot] == 0);
                prices_[slot] = kEmpty;           // masks it out of all scans
                free_.push_back(slot);
                --live_;
            }
        }
    }

    // ---- the on-demand query --------------------------------------------

    // Best price such that cumulative aggregate size from the inside down
    // (bids: descending; asks: ascending) first reaches round_lot.
    // nullopt if the whole side sums to less than round_lot.
    [[nodiscard]] std::optional<RoundLotBest>
    roundlot_best(std::uint32_t round_lot) const noexcept {
        std::uint64_t cum = 0;
        std::uint32_t used_levels = 0;
        std::int32_t  bound = kBid ? std::numeric_limits<std::int32_t>::max()
                                   : std::numeric_limits<std::int32_t>::min();
        for (;;) {
            const std::int32_t idx = scan_best(bound);
            if (idx < 0) return std::nullopt;     // side exhausted below lot
            const auto u = static_cast<std::size_t>(idx);
            cum += sizes_[u];
            ++used_levels;
            if (cum >= round_lot)
                return RoundLotBest{prices_[u], cum, used_levels};
            bound = prices_[u];                   // exclude this and better
        }
    }

    // Plain best (top of side) — a single scan.
    [[nodiscard]] std::optional<std::int32_t> best() const noexcept {
        const std::int32_t idx = scan_best(
            kBid ? std::numeric_limits<std::int32_t>::max()
                 : std::numeric_limits<std::int32_t>::min());
        if (idx < 0) return std::nullopt;
        return prices_[static_cast<std::size_t>(idx)];
    }

    [[nodiscard]] std::size_t   live_levels() const noexcept { return live_; }
    [[nodiscard]] std::size_t   slots()       const noexcept { return used_; }
    [[nodiscard]] std::int32_t  price_at(std::uint32_t s) const noexcept { return prices_[s]; }
    [[nodiscard]] std::uint32_t size_at (std::uint32_t s) const noexcept { return sizes_[s]; }

private:
    // Best price strictly worse-than-bound for this side, among live slots:
    // bids: max price < bound; asks: min price > bound. Returns slot or -1.
    [[nodiscard]] std::int32_t scan_best(std::int32_t bound) const noexcept {
        const std::size_t n = used_;
#if defined(__AVX2__)
        const __m256i vbound = _mm256_set1_epi32(bound);
        const __m256i vempty = _mm256_set1_epi32(kEmpty);
        __m256i vbest = vempty;
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(prices_.data() + i));
            // keep only entries strictly inside the bound; others -> kEmpty
            __m256i ok = kBid ? _mm256_cmpgt_epi32(vbound, v)   // v < bound
                              : _mm256_cmpgt_epi32(v, vbound);  // v > bound
            v = _mm256_blendv_epi8(vempty, v, ok);
            vbest = kBid ? _mm256_max_epi32(vbest, v)
                         : _mm256_min_epi32(vbest, v);
        }
        // horizontal reduce of vbest
        alignas(32) std::int32_t lanes[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), vbest);
        std::int32_t best = kEmpty;
        for (int l = 0; l < 8; ++l)
            best = kBid ? (lanes[l] > best ? lanes[l] : best)
                        : (lanes[l] < best ? lanes[l] : best);
        for (; i < n; ++i) {                       // scalar tail
            const std::int32_t p = prices_[i];
            if (p == kEmpty) continue;
            if (kBid ? (p < bound && p > best) : (p > bound && p < best)) best = p;
        }
        if (best == kEmpty) return -1;
        // locate the winning slot: SIMD equality scan + movemask
        const __m256i vb = _mm256_set1_epi32(best);
        for (std::size_t j = 0; j + 8 <= n; j += 8) {
            __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(prices_.data() + j));
            const int m = _mm256_movemask_ps(
                _mm256_castsi256_ps(_mm256_cmpeq_epi32(v, vb)));
            if (m) return static_cast<std::int32_t>(
                j + static_cast<unsigned>(std::countr_zero(
                        static_cast<unsigned>(m))));
        }
        for (std::size_t j = n & ~std::size_t{7}; j < n; ++j)
            if (prices_[j] == best) return static_cast<std::int32_t>(j);
        return -1;                                 // unreachable if best found
#else
        std::int32_t best = kEmpty; std::int32_t at = -1;
        for (std::size_t i = 0; i < n; ++i) {
            const std::int32_t p = prices_[i];
            if (p == kEmpty) continue;
            if (kBid ? (p < bound && p > best) : (p > bound && p < best)) {
                best = p; at = static_cast<std::int32_t>(i);
            }
        }
        return at;
#endif
    }

    // Equality scan for an existing level's slot (add path). -1 if absent.
    [[nodiscard]] std::int32_t find_price(std::int32_t prc) const noexcept {
        const std::size_t n = used_;
#if defined(__AVX2__)
        const __m256i vp = _mm256_set1_epi32(prc);
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(prices_.data() + i));
            const int m = _mm256_movemask_ps(
                _mm256_castsi256_ps(_mm256_cmpeq_epi32(v, vp)));
            if (m) return static_cast<std::int32_t>(
                i + static_cast<unsigned>(std::countr_zero(
                        static_cast<unsigned>(m))));
        }
        for (; i < n; ++i)
            if (prices_[i] == prc) return static_cast<std::int32_t>(i);
        return -1;
#else
        for (std::size_t i = 0; i < n; ++i)
            if (prices_[i] == prc) return static_cast<std::int32_t>(i);
        return -1;
#endif
    }

    static std::size_t round_up8(std::size_t v) noexcept { return (v + 7) & ~std::size_t{7}; }

    void grow() {
        const std::size_t cap = prices_.size() * 2;
        prices_.resize(cap, kEmpty);   // stable indices: resize, never reorder.
        sizes_.resize(cap, 0);         // NOTE: resize may reallocate — pre-size
        norders_.resize(cap, 0);       // reserve_levels past worst depth so this
    }                                  // never runs mid-session.

    std::vector<std::int32_t>  prices_;   // kEmpty = tombstone/unused
    std::vector<std::uint32_t> sizes_;
    std::vector<std::uint32_t> norders_;
    std::vector<std::uint32_t> free_;     // tombstoned slots for reuse
    std::size_t used_ = 0;                // high-water slot count
    std::size_t live_ = 0;                // live (non-tombstone) levels
};

}  // namespace md
