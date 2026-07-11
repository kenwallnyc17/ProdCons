// sorted_index_book.hpp — price-level book with in-order traversal:
// a SORTED INDEX over an UNSORTED HEAP.
//
//   * Level data (agg size, order count) lives in stable unsorted SoA slots,
//     exactly as before: Order.lvl -> slot, O(1) reduce, tombstone+freelist.
//   * Ordering lives in two parallel sorted arrays (SoA, ascending price):
//       sorted_prc_[i]  : int32 price
//       sorted_idx_[i]  : uint32 slot into the unsorted arrays
//     (the user's PrcIndex{prc, idx} split into SoA: the lower_bound search
//     touches only prices, so SoA doubles search density per cache line;
//     memmove bytes on insert are the same either way.)
//
// The invariant that makes it cheap: the sorted arrays change ONLY on level
// create/destroy. Size updates — the vast majority of messages — go through
// Order.lvl and never touch the index. A liquid book mutates sizes millions
// of times but creates/destroys levels far less often.
//
// One lower_bound serves the whole Add path: hit -> existing slot; miss ->
// the insertion position for the memmove. Search = branchless binary search
// narrowed to a 32-element window, then a branch-free AVX2 count within it.
//
// roundlot_best / traversal: walk the sorted arrays from the inside end
// (bids: back->front, asks: front->back), accumulating sizes_[slot]. O(k) for
// k levels consumed; no scans, no masks. The SIMD max-scan design is retired.
//
// Honest cost: level insert/erase memmoves the tail of both arrays — O(n)
// each, ~4 bytes/entry/array. At n=1024 that is ~4KB average per array, well
// under a microsecond, and only on create/destroy. The tail case is a sweep
// destroying k levels: k memmoves. If profiling shows that matters, the
// mitigation is lazy deletion (an index entry is stale iff
// prices_[sorted_idx_[i]] != sorted_prc_[i]; traversal skips stale, compact
// periodically) — not implemented here, eager is simpler and usually fine.
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

enum class SSide : std::uint8_t { Bid = 0, Ask = 1 };

struct SRlBest {
    std::int32_t  price;
    std::uint64_t cum_size;
    std::uint32_t levels_used;
};

template <SSide Side>
class SortedIndexBook {
    static constexpr bool kBid = (Side == SSide::Bid);
    static constexpr std::int32_t kTomb = std::numeric_limits<std::int32_t>::min();

public:
    explicit SortedIndexBook(std::size_t max_levels = 1024) {
        const std::size_t cap = max_levels ? max_levels : 8;
        prices_.assign(cap, kTomb);
        sizes_.assign(cap, 0);
        norders_.assign(cap, 0);
        sorted_prc_.assign(cap, 0);
        sorted_idx_.assign(cap, 0);
    }

    // ---- hot paths --------------------------------------------------------

    // Add order flow; creates the level if absent. Returns stable slot index
    // for Order.lvl. One lower_bound answers exists/insert-position both.
    std::uint32_t add(std::int32_t prc, std::uint32_t sz) noexcept {
        const std::size_t pos = lower_bound(prc);
        if (pos < scount_ && sorted_prc_[pos] == prc) {      // existing level
            const std::uint32_t s = sorted_idx_[pos];
            sizes_[s]   += sz;
            norders_[s] += 1;
            return s;
        }
        // create: slot from freelist/high-water, splice into sorted arrays
        std::uint32_t slot;
        if (!free_.empty()) { slot = free_.back(); free_.pop_back(); }
        else {
            assert(used_ < prices_.size() && "max_levels too small");
            slot = static_cast<std::uint32_t>(used_++);
        }
        prices_[slot]  = prc;
        sizes_[slot]   = sz;
        norders_[slot] = 1;

        const std::size_t tail = scount_ - pos;
        std::memmove(&sorted_prc_[pos + 1], &sorted_prc_[pos], tail * sizeof(std::int32_t));
        std::memmove(&sorted_idx_[pos + 1], &sorted_idx_[pos], tail * sizeof(std::uint32_t));
        sorted_prc_[pos] = prc;
        sorted_idx_[pos] = slot;
        ++scount_;
        return slot;
    }

    // Reduce by slot (cancel/execute: Order.lvl in hand). Sorted index is
    // touched only if the level dies.
    void reduce(std::uint32_t slot, std::uint32_t sz, bool order_gone) noexcept {
        assert(slot < used_ && prices_[slot] != kTomb);
        assert(sizes_[slot] >= sz);
        sizes_[slot] -= sz;
        if (order_gone && --norders_[slot] == 0) {
            assert(sizes_[slot] == 0);
            const std::size_t pos = lower_bound(prices_[slot]);
            assert(pos < scount_ && sorted_prc_[pos] == prices_[slot]);
            const std::size_t tail = scount_ - pos - 1;
            std::memmove(&sorted_prc_[pos], &sorted_prc_[pos + 1], tail * sizeof(std::int32_t));
            std::memmove(&sorted_idx_[pos], &sorted_idx_[pos + 1], tail * sizeof(std::uint32_t));
            --scount_;
            prices_[slot] = kTomb;
            free_.push_back(slot);
        }
    }

    // ---- in-order access (the point of this variant) ------------------------

    // Visit levels from the inside outward: f(price, agg_size, num_orders).
    // Return false from f to stop early.
    template <class F>
    void for_each_from_best(F&& f) const
        noexcept(noexcept(f(std::int32_t{}, std::uint32_t{}, std::uint32_t{}))) {
        if constexpr (kBid) {
            for (std::size_t i = scount_; i-- > 0;) {
                const std::uint32_t s = sorted_idx_[i];
                if (!f(sorted_prc_[i], sizes_[s], norders_[s])) return;
            }
        } else {
            for (std::size_t i = 0; i < scount_; ++i) {
                const std::uint32_t s = sorted_idx_[i];
                if (!f(sorted_prc_[i], sizes_[s], norders_[s])) return;
            }
        }
    }

    [[nodiscard]] std::optional<SRlBest>
    roundlot_best(std::uint32_t round_lot) const noexcept {
        std::uint64_t cum = 0;
        std::uint32_t used_levels = 0;
        std::optional<SRlBest> out;
        for_each_from_best([&](std::int32_t p, std::uint32_t s, std::uint32_t) {
            cum += s;
            ++used_levels;
            if (cum >= round_lot) { out = SRlBest{p, cum, used_levels}; return false; }
            return true;
        });
        return out;
    }

    [[nodiscard]] std::optional<std::int32_t> best() const noexcept {
        if (scount_ == 0) return std::nullopt;
        return kBid ? sorted_prc_[scount_ - 1] : sorted_prc_[0];
    }

    // nth best level (0 = inside): (price, slot) — for top-N snapshots.
    [[nodiscard]] std::optional<std::pair<std::int32_t, std::uint32_t>>
    nth_best(std::size_t n) const noexcept {
        if (n >= scount_) return std::nullopt;
        const std::size_t i = kBid ? scount_ - 1 - n : n;
        return std::pair{sorted_prc_[i], sorted_idx_[i]};
    }

    [[nodiscard]] std::size_t   live_levels() const noexcept { return scount_; }
    [[nodiscard]] std::size_t   slots()       const noexcept { return used_; }
    [[nodiscard]] std::uint32_t size_at (std::uint32_t s) const noexcept { return sizes_[s]; }
    [[nodiscard]] std::int32_t  price_at(std::uint32_t s) const noexcept { return prices_[s]; }

private:
    // First position with sorted_prc_[pos] >= prc. Branchless binary search
    // down to a 32-element window, then branch-free SIMD count within it.
    [[nodiscard]] std::size_t lower_bound(std::int32_t prc) const noexcept {
        const std::int32_t* a = sorted_prc_.data();
        std::size_t lo = 0, len = scount_;
        while (len > 32) {                       // branchless halving (cmov)
            const std::size_t half = len / 2;
            const std::size_t mid  = lo + half;
            lo  = (a[mid] < prc) ? mid : lo;     // keep the half containing pos
            len -= half;
        }
#if defined(__AVX2__)
        // count entries < prc in a[lo, lo+len); len <= 32 -> at most 4 vectors
        const __m256i vk = _mm256_set1_epi32(prc);
        std::size_t cnt = 0, i = 0;
        for (; i + 8 <= len; i += 8) {
            __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(a + lo + i));
            const unsigned m = static_cast<unsigned>(_mm256_movemask_ps(
                _mm256_castsi256_ps(_mm256_cmpgt_epi32(vk, v))));  // v < prc
            cnt += static_cast<unsigned>(std::popcount(m));
        }
        for (; i < len; ++i) cnt += (a[lo + i] < prc);
        return lo + cnt;
#else
        std::size_t cnt = 0;
        for (std::size_t i = 0; i < len; ++i) cnt += (a[lo + i] < prc);
        return lo + cnt;
#endif
    }

    // unsorted stable heap (Order.lvl targets; kTomb marks freed slots)
    std::vector<std::int32_t>  prices_;
    std::vector<std::uint32_t> sizes_;
    std::vector<std::uint32_t> norders_;
    std::vector<std::uint32_t> free_;
    std::size_t used_ = 0;

    // sorted index (ascending price; bids read back->front)
    std::vector<std::int32_t>  sorted_prc_;
    std::vector<std::uint32_t> sorted_idx_;
    std::size_t scount_ = 0;
};

}  // namespace md
