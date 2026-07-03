// static_bucketed_order_id_map.hpp — order-reference -> record map with a
// COMPILE-TIME directory and a PRE-SIZED overflow pool. Zero runtime
// allocation on any path; grows without rehash.
//
// Storage model (the point of this variant):
//   * The directory is Block dir_[1<<BucketBits], a fixed-size member array.
//     Declare the map in static storage and it is reserved at link time and
//     demand-zero committed by the loader — "created at compile", no malloc.
//   * Overflow blocks come from Block pool_[PoolBlocks], a fixed member pool
//     handed out by a bump cursor then a freelist. When the pool is exhausted
//     the map does NOT allocate — upsert returns PoolExhausted so the caller
//     can alarm. Size the pool for your worst case; exhaustion is a config bug.
//
// Empty-tag sentinel is 0, so a zero-initialised block is already an empty
// bucket: no init pass over the (multi-GB) directory at startup, pages commit
// lazily on first touch. Fragments are mapped into 1..127 so they never alias
// the empty tag. Deletion is swap-remove (no tombstones); an emptied block has
// all-empty tags by construction, so pool reuse is safe.
//
// REQUIRED: the object must live in zero-initialised storage — a global/static
// (ideally `constinit`), or placement-new into a MAP_ANONYMOUS / MAP_HUGETLB
// mapping (also zero-filled). Do NOT create it on the stack; it is huge and its
// emptiness depends on zero storage. It is non-copyable and non-movable.
//
// Single writer. No atomics/locks inside. Build: -std=c++23 -O3 -march=native.

#pragma once

#include "swiss_order_id_map.hpp"   // OrderRecord + Group{Scalar,Avx2,Avx512}

namespace md {

enum class Put : std::uint8_t { Inserted, Updated, PoolExhausted };

template <class Value = OrderRecord, class Group = DefaultGroup,
          std::size_t BucketBits = 22,            // 1<<22 = 4M buckets
          std::size_t PoolBlocks = (1u << 20),    // 1M overflow blocks
          std::uint64_t Empty = 0>
class StaticBucketedOrderIdMap {
    static_assert(std::is_trivially_copyable_v<Value>);
    static_assert(BucketBits >= 7 && BucketBits <= 57);
    static constexpr std::size_t CAP = Group::kWidth;     // 16 / 32 / 64
    using Mask = typename Group::Mask;

public:
    using key_type   = std::uint64_t;
    using value_type = Value;
    static constexpr std::size_t kBuckets  = std::size_t{1} << BucketBits;
    static constexpr std::size_t kPool     = PoolBlocks;
    static constexpr std::size_t kCap      = CAP;
    static constexpr key_type    kEmptyKey = Empty;
    static constexpr std::int8_t kTagEmpty = 0;           // zero-init == empty

    struct alignas(64) Block {
        std::int8_t  tags[CAP];   // 0 = empty, 1..127 = fragment
        std::uint32_t count;      // dense occupancy in [0, count)
        Block*       next;        // overflow chain
        key_type     keys[CAP];
        Value        vals[CAP];
    };

    // Trivial default constructor: writes NOTHING. A zeroed StaticBucketed map
    // IS a valid empty map (empty tag 0, count 0, next null, all cursors 0), so
    // it must live in zero-initialised storage — static/BSS (zero-filled at
    // load) or an mmap'd region (kernel-zero-filled). This is what lets the
    // hugepage placement factory construct in-place without touching the
    // multi-GB directory: no giant memset, pages commit lazily / on populate.
    StaticBucketedOrderIdMap() noexcept = default;

    StaticBucketedOrderIdMap(const StaticBucketedOrderIdMap&)            = delete;
    StaticBucketedOrderIdMap& operator=(const StaticBucketedOrderIdMap&) = delete;

    // ---- hot path -------------------------------------------------------

    [[nodiscard]] Value* find(key_type key) noexcept {
        assert(key != kEmptyKey);
        const std::uint64_t h = mix(key);
        const std::int8_t   h2 = frag(h);
        Block* blk = &dir_[h >> kDirShift];
        do {
            Mask m = Group::match(blk->tags, h2);
            while (m) {
                const unsigned i = static_cast<unsigned>(std::countr_zero(m));
                if (blk->keys[i] == key) [[likely]] return &blk->vals[i];
                m &= m - 1;
            }
            blk = blk->next;
        } while (blk);
        return nullptr;
    }

    [[nodiscard]] const Value* find(key_type key) const noexcept {
        return const_cast<StaticBucketedOrderIdMap*>(this)->find(key);
    }

    // Insert or overwrite. Never rehashes, never calls the allocator. On a full
    // chain it pulls one block from the fixed pool; if the pool is empty it
    // returns PoolExhausted (and bumps a counter) rather than dropping silently.
    Put upsert(key_type key, Value value) noexcept {
        assert(key != kEmptyKey);
        const std::uint64_t h = mix(key);
        const std::int8_t   h2 = frag(h);
        Block* const head = &dir_[h >> kDirShift];

        Block* room = nullptr;
        for (Block* blk = head;; blk = blk->next) {
            Mask m = Group::match(blk->tags, h2);
            while (m) {
                const unsigned i = static_cast<unsigned>(std::countr_zero(m));
                if (blk->keys[i] == key) { blk->vals[i] = value; return Put::Updated; }
                m &= m - 1;
            }
            if (!room && blk->count < CAP) room = blk;
            if (!blk->next) break;
        }
        if (!room) {
            Block* nb = pool_alloc();
            if (!nb) [[unlikely]] { ++insert_failures_; return Put::PoolExhausted; }
            nb->next = head->next;
            head->next = nb;
            room = nb;
        }
        const unsigned i = room->count++;
        room->tags[i] = h2;
        room->keys[i] = key;
        room->vals[i] = value;
        ++size_;
        return Put::Inserted;
    }

    // Erase by key (swap-remove). Returns removed payload via *out. Reclaims an
    // emptied overflow block back to the pool.
    bool erase(key_type key, Value* out = nullptr) noexcept {
        assert(key != kEmptyKey);
        const std::uint64_t h = mix(key);
        const std::int8_t   h2 = frag(h);
        Block* const head = &dir_[h >> kDirShift];
        Block* prev = nullptr;
        for (Block* blk = head; blk; prev = blk, blk = blk->next) {
            Mask m = Group::match(blk->tags, h2);
            while (m) {
                const unsigned i = static_cast<unsigned>(std::countr_zero(m));
                if (blk->keys[i] == key) {
                    if (out) *out = blk->vals[i];
                    const unsigned last = --blk->count;
                    blk->tags[i] = blk->tags[last];
                    blk->keys[i] = blk->keys[last];
                    blk->vals[i] = blk->vals[last];
                    blk->tags[last] = kTagEmpty;
                    --size_;
                    if (blk != head && blk->count == 0) {
                        prev->next = blk->next;
                        pool_free(blk);
                    }
                    return true;
                }
                m &= m - 1;
            }
        }
        return false;
    }

    // ---- lifecycle / introspection -------------------------------------

    // Commit and warm every directory and pool page before the open so the
    // first real sweep is not a page fault. Touch one byte per 4K page.
    void prefault() noexcept {
        auto touch = [](void* base, std::size_t bytes) {
            volatile char* p = static_cast<volatile char*>(base);
            for (std::size_t off = 0; off < bytes; off += 4096) p[off] = p[off];
        };
        touch(dir_, sizeof(dir_));
        touch(pool_, sizeof(pool_));
    }

    [[nodiscard]] std::size_t size()            const noexcept { return size_; }
    [[nodiscard]] static constexpr std::size_t num_buckets() noexcept { return kBuckets; }
    [[nodiscard]] static constexpr std::size_t pool_blocks() noexcept { return kPool; }
    [[nodiscard]] static constexpr std::size_t bucket_cap()  noexcept { return CAP; }
    [[nodiscard]] static constexpr std::size_t block_bytes() noexcept { return sizeof(Block); }
    [[nodiscard]] static constexpr std::size_t footprint_bytes() noexcept {
        return sizeof(Block) * (kBuckets + kPool);
    }
    [[nodiscard]] std::size_t pool_in_use()     const noexcept {
        return pool_high_ - pool_free_count_;
    }
    [[nodiscard]] std::size_t insert_failures() const noexcept { return insert_failures_; }
    [[nodiscard]] double      avg_depth()       const noexcept {
        return static_cast<double>(size_) / static_cast<double>(kBuckets);
    }
    [[nodiscard]] std::size_t max_chain() const noexcept {
        std::size_t worst = 1;
        for (std::size_t b = 0; b < kBuckets; ++b) {
            std::size_t n = 1;
            for (Block* x = dir_[b].next; x; x = x->next) ++n;
            worst = worst < n ? n : worst;
        }
        return worst;
    }

private:
    static constexpr std::uint64_t kFib      = 0x9E3779B97F4A7C15ULL;
    static constexpr unsigned      kDirShift = 64u - static_cast<unsigned>(BucketBits);

    static std::uint64_t mix(key_type key) noexcept { return key * kFib; }
    static std::int8_t frag(std::uint64_t h) noexcept {
        std::uint8_t f = static_cast<std::uint8_t>((h >> (kDirShift - 7)) & 0x7F);
        f = static_cast<std::uint8_t>(f + (f == 0));   // 0 -> 1, keep top bit clear
        return static_cast<std::int8_t>(f);
    }

    Block* pool_alloc() noexcept {
        if (free_head_) {                       // recycle a returned block
            Block* b = free_head_;
            free_head_ = b->next;
            b->next = nullptr;
            b->count = 0;
            --pool_free_count_;
            return b;
        }
        if (pool_high_ < kPool) {               // carve a fresh (zeroed) block
            return &pool_[pool_high_++];        // already empty: zero storage
        }
        return nullptr;                         // pool exhausted — sized wrong
    }
    void pool_free(Block* b) noexcept {
        std::memset(b->tags, kTagEmpty, CAP);   // insurance; invariant already empties it
        b->count = 0;
        b->next  = free_head_;
        free_head_ = b;
        ++pool_free_count_;
    }

    // Fixed storage. Lives in zero-initialised static (or anonymous-mmap) memory.
    alignas(64) Block dir_[kBuckets];
    Block             pool_[kPool];
    Block*       free_head_;
    std::size_t  pool_high_;        // bump cursor into pool_
    std::size_t  pool_free_count_;  // blocks currently on the freelist
    std::size_t  size_;
    std::size_t  insert_failures_;
};

}  // namespace md
