// swiss_order_id_map.hpp — single-writer order-reference -> record map for an
// L3 equity feed handler, with SIMD group probing (SwissTable/F14 style).
//
// Where SIMD is "necessary": the probe. A 1-byte control tag per slot holds a
// 7-bit hash fragment; a group of W tags (W = 32 on AVX2, 64 on AVX-512BW) is
// scanned in a single vector compare, so a lookup is one aligned load + one
// cmpeq + one mask -> iterate only the handful of fragment matches, verifying
// the full 8-byte key just for those. Non-matches never touch the key array.
//
// This replaces the previous header's Robin Hood backward-shift with tag-based
// probing. Deletion therefore reintroduces tombstones — but with group-aligned
// probing we mark a slot EMPTY (not DELETED) whenever its group still holds an
// empty slot, which at load <= 0.5 is almost always. So real tombstones stay
// rare and probe lengths do not creep over a churning session.
//
//   * Group-aligned probing => all SIMD loads are aligned, no mirror region.
//   * Triangular re-probe over groups: visits every group (power-of-two count).
//   * Single 64-bit multiply hash; H1 (group) and H2 (fragment) both taken from
//     the well-mixed high bits.
//   * Single writer. No atomics/locks inside.
//
// Build: -std=c++23 -O3 -march=native   (auto-selects the widest ISA present).
// For one portable binary across CPUs, compile the group policy methods with
// target_clones / __attribute__((target(...))) and dispatch on
// __builtin_cpu_supports — omitted here since latency handlers pin to a CPU.

#pragma once

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>

#if defined(__AVX2__) || defined(__AVX512BW__)
#  include <immintrin.h>
#endif
#if defined(__linux__)
#  include <sys/mman.h>
#endif

namespace md {

struct OrderRecord {
    std::uint32_t level_idx = 0;  // index into the symbol's paged ladder
    std::uint32_t qty       = 0;  // remaining shares on this order
    std::uint16_t book_id   = 0;  // which symbol/book
    std::uint8_t  side      = 0;  // 0 = bid, 1 = ask
    std::uint8_t  _pad      = 0;
};

// Control-tag values. Full slots store a 7-bit fragment with the top bit clear
// (0..127); empty/deleted have the top bit set, so one sign-bit movemask finds
// every non-full slot at once.
inline constexpr std::int8_t kCtrlEmpty   = static_cast<std::int8_t>(0x80); // -128
inline constexpr std::int8_t kCtrlDeleted = static_cast<std::int8_t>(0xFE); // -2

// ---- group policies ------------------------------------------------------
// Each exposes: kWidth, Mask, and three scans over W aligned control bytes.
//   match(ctrl,h2): tags equal to fragment h2
//   match_empty(ctrl): tags equal to EMPTY
//   mask_free(ctrl): tags that are EMPTY or DELETED (top bit set)

struct GroupScalar {
    static constexpr std::size_t kWidth = 16;
    using Mask = std::uint32_t;
    static Mask match(const std::int8_t* c, std::int8_t h2) noexcept {
        Mask m = 0;
        for (std::size_t i = 0; i < kWidth; ++i) m |= Mask(c[i] == h2) << i;
        return m;
    }
    static Mask match_empty(const std::int8_t* c) noexcept {
        Mask m = 0;
        for (std::size_t i = 0; i < kWidth; ++i) m |= Mask(c[i] == kCtrlEmpty) << i;
        return m;
    }
    static Mask mask_free(const std::int8_t* c) noexcept {
        Mask m = 0;
        for (std::size_t i = 0; i < kWidth; ++i) m |= Mask(c[i] < 0) << i; // top bit
        return m;
    }
};

#if defined(__AVX2__)
struct GroupAvx2 {
    static constexpr std::size_t kWidth = 32;
    using Mask = std::uint32_t;
    static __m256i load(const std::int8_t* c) noexcept {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(c));
    }
    static Mask match(const std::int8_t* c, std::int8_t h2) noexcept {
        __m256i v = load(c);
        return static_cast<Mask>(_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(v, _mm256_set1_epi8(h2))));
    }
    static Mask match_empty(const std::int8_t* c) noexcept {
        __m256i v = load(c);
        return static_cast<Mask>(_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(v, _mm256_set1_epi8(kCtrlEmpty))));
    }
    static Mask mask_free(const std::int8_t* c) noexcept {
        // full tags have top bit 0; empty/deleted have top bit 1
        return static_cast<Mask>(_mm256_movemask_epi8(load(c)));
    }
};
#endif

#if defined(__AVX512BW__)
struct GroupAvx512 {
    static constexpr std::size_t kWidth = 64;
    using Mask = std::uint64_t;
    static __m512i load(const std::int8_t* c) noexcept {
        return _mm512_load_si512(reinterpret_cast<const void*>(c));
    }
    static Mask match(const std::int8_t* c, std::int8_t h2) noexcept {
        return _mm512_cmpeq_epi8_mask(load(c), _mm512_set1_epi8(h2));
    }
    static Mask match_empty(const std::int8_t* c) noexcept {
        return _mm512_cmpeq_epi8_mask(load(c), _mm512_set1_epi8(kCtrlEmpty));
    }
    static Mask mask_free(const std::int8_t* c) noexcept {
        return _mm512_movepi8_mask(load(c)); // top bit of each byte
    }
};
#endif

// Pick the widest available group at compile time.
#if defined(__AVX512BW__)
using DefaultGroup = GroupAvx512;
#elif defined(__AVX2__)
using DefaultGroup = GroupAvx2;
#else
using DefaultGroup = GroupScalar;
#endif

template <class Value = OrderRecord, class Group = DefaultGroup,
          std::uint64_t Empty = 0>
class SwissOrderIdMap {
    static_assert(std::is_trivially_copyable_v<Value>);
    static constexpr std::size_t W = Group::kWidth;
    using Mask = typename Group::Mask;

public:
    using key_type   = std::uint64_t;
    using value_type = Value;
    static constexpr key_type kEmptyKey = Empty;

    struct Slot {
        key_type ref;
        Value    rec;
    };

    explicit SwissOrderIdMap(std::size_t peak_resident, double max_load = 0.5) {
        assert(max_load > 0.0 && max_load < 1.0);
        const std::size_t need =
            static_cast<std::size_t>(static_cast<double>(peak_resident) / max_load) + 1;
        std::size_t groups = (need + W - 1) / W;
        num_groups_ = std::bit_ceil(groups < 1 ? std::size_t{1} : groups);
        group_mask_ = num_groups_ - 1;
        capacity_   = num_groups_ * W;
        // group index is taken from the top bits of the mixed hash
        g_shift_    = 64u - static_cast<unsigned>(std::countr_zero(num_groups_));

        ctrl_  = static_cast<std::int8_t*>(aligned_alloc_bytes(capacity_, 64));
        slots_ = static_cast<Slot*>(aligned_alloc_bytes(capacity_ * sizeof(Slot), 64));
        clear();
    }

    ~SwissOrderIdMap() {
        free_bytes(ctrl_);
        free_bytes(slots_);
    }

    SwissOrderIdMap(const SwissOrderIdMap&)            = delete;
    SwissOrderIdMap& operator=(const SwissOrderIdMap&) = delete;

    // ---- hot path -------------------------------------------------------

    [[nodiscard]] Value* find(key_type key) noexcept {
        assert(key != kEmptyKey);
        const std::uint64_t h = mix(key);
        const std::int8_t   h2 = frag(h);
        std::size_t g = h >> g_shift_;
        std::size_t step = 0;
        for (;;) {
            const std::size_t base = g * W;
            const std::int8_t* c = ctrl_ + base;
            Mask m = Group::match(c, h2);
            while (m) {
                const unsigned i = static_cast<unsigned>(std::countr_zero(m));
                if (slots_[base + i].ref == key) [[likely]] return &slots_[base + i].rec;
                m &= m - 1;
            }
            if (Group::match_empty(c)) return nullptr;  // empty in group => absent
            step += 1;
            g = (g + step) & group_mask_;               // triangular probe
        }
    }

    [[nodiscard]] const Value* find(key_type key) const noexcept {
        return const_cast<SwissOrderIdMap*>(this)->find(key);
    }

    // Insert or overwrite. Returns true if a new key was inserted.
    bool upsert(key_type key, Value value) noexcept {
        assert(key != kEmptyKey);
        assert(size_ < capacity_ && "table full — sized too small to never rehash");
        const std::uint64_t h = mix(key);
        const std::int8_t   h2 = frag(h);
        std::size_t g = h >> g_shift_;
        std::size_t step = 0;

        // Track the first reusable (empty-or-deleted) slot along the probe so a
        // tombstone gets reclaimed rather than skipped.
        bool have_free = false;
        std::size_t free_base = 0;
        unsigned    free_i    = 0;

        for (;;) {
            const std::size_t base = g * W;
            const std::int8_t* c = ctrl_ + base;

            Mask mm = Group::match(c, h2);
            while (mm) {
                const unsigned i = static_cast<unsigned>(std::countr_zero(mm));
                if (slots_[base + i].ref == key) {  // existing -> update in place
                    slots_[base + i].rec = value;
                    return false;
                }
                mm &= mm - 1;
            }

            if (!have_free) {
                Mask freem = Group::mask_free(c);     // empty or deleted
                if (freem) {
                    have_free = true;
                    free_base = base;
                    free_i    = static_cast<unsigned>(std::countr_zero(freem));
                }
            }

            if (Group::match_empty(c)) {              // absence proven; place it
                const std::size_t b = have_free ? free_base : base;
                const unsigned    i = have_free
                    ? free_i
                    : static_cast<unsigned>(std::countr_zero(Group::match_empty(c)));
                ctrl_[b + i]      = h2;
                slots_[b + i].ref = key;
                slots_[b + i].rec = value;
                ++size_;
                return true;
            }
            step += 1;
            g = (g + step) & group_mask_;
        }
    }

    // Erase by key. Copies the removed payload into *out (qty to hand back to
    // the ladder). Marks the slot EMPTY when its group still has an empty slot
    // (no tombstone), else DELETED.
    bool erase(key_type key, Value* out = nullptr) noexcept {
        assert(key != kEmptyKey);
        const std::uint64_t h = mix(key);
        const std::int8_t   h2 = frag(h);
        std::size_t g = h >> g_shift_;
        std::size_t step = 0;
        for (;;) {
            const std::size_t base = g * W;
            const std::int8_t* c = ctrl_ + base;
            Mask m = Group::match(c, h2);
            while (m) {
                const unsigned i = static_cast<unsigned>(std::countr_zero(m));
                if (slots_[base + i].ref == key) {
                    if (out) *out = slots_[base + i].rec;
                    ctrl_[base + i] = Group::match_empty(c) ? kCtrlEmpty : kCtrlDeleted;
                    slots_[base + i].ref = kEmptyKey;
                    --size_;
                    return true;
                }
                m &= m - 1;
            }
            if (Group::match_empty(c)) return false;
            step += 1;
            g = (g + step) & group_mask_;
        }
    }

    // ---- lifecycle / introspection -------------------------------------

    void clear() noexcept {
        std::memset(ctrl_, static_cast<int>(static_cast<unsigned char>(kCtrlEmpty)),
                    capacity_);
        for (std::size_t i = 0; i < capacity_; ++i) slots_[i].ref = kEmptyKey;
        size_ = 0;
    }

    // Commit pages before the open so the first real sweep isn't a page fault.
    void prefault() noexcept { clear(); }

    [[nodiscard]] std::size_t size()       const noexcept { return size_; }
    [[nodiscard]] std::size_t capacity()   const noexcept { return capacity_; }
    [[nodiscard]] std::size_t group_width()const noexcept { return W; }
    [[nodiscard]] double      load()       const noexcept {
        return static_cast<double>(size_) / static_cast<double>(capacity_);
    }
    [[nodiscard]] std::size_t tombstones() const noexcept {
        std::size_t n = 0;
        for (std::size_t i = 0; i < capacity_; ++i) n += (ctrl_[i] == kCtrlDeleted);
        return n;
    }

private:
    static constexpr std::uint64_t kFib = 0x9E3779B97F4A7C15ULL;

    // Single multiply; high bits are best-mixed, so both the group index
    // (h >> g_shift_) and the fragment come from the top of the word.
    static std::uint64_t mix(key_type key) noexcept { return key * kFib; }
    std::int8_t frag(std::uint64_t h) const noexcept {
        // 7 bits just below the group-index bits; guaranteed top bit clear.
        return static_cast<std::int8_t>((h >> (g_shift_ - 7)) & 0x7F);
    }

    static void* aligned_alloc_bytes(std::size_t bytes, std::size_t align) {
        void* p = nullptr;
#if defined(__linux__)
        const std::size_t huge = 2 * 1024 * 1024;
        const std::size_t a = align < huge ? huge : align;     // 2MB for THP
        if (posix_memalign(&p, a, bytes) != 0) p = nullptr;
        if (p) ::madvise(p, bytes, MADV_HUGEPAGE);
#else
        p = ::operator new(bytes, std::align_val_t{align});
#endif
        if (!p) throw std::bad_alloc{};
        return p;
    }
    static void free_bytes(void* p) noexcept {
        if (!p) return;
#if defined(__linux__)
        std::free(p);
#else
        ::operator delete(p);
#endif
    }

    std::int8_t* ctrl_       = nullptr;   // capacity_ tags, 64B aligned
    Slot*        slots_      = nullptr;   // capacity_ slots, 64B aligned
    std::size_t  num_groups_ = 0;
    std::size_t  group_mask_ = 0;
    std::size_t  capacity_   = 0;
    std::size_t  size_       = 0;
    unsigned     g_shift_    = 0;
};

}  // namespace md
