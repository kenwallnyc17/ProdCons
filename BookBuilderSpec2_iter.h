#ifndef BOOKBUILDERSPEC2_ITER_H_INCLUDED
#define BOOKBUILDERSPEC2_ITER_H_INCLUDED
/*
#include <cstdint>
#include <array>
#include <iterator>
#include <iostream>
//#include <numeric>
#include <cassert>
#include <algorithm>
#include <utility>
#include <concepts>

//#include <pair>

using namespace std;
*/
constexpr uint32_t NIL = 0;

// High-frequency trading market data fields
using PriceType       = uint64_t;
using SizeType        = uint32_t;
using NumOrdersType   = uint16_t;
using FeedTimeType    = uint64_t;
using UniqueUpdateID  = uint64_t;

struct PrcLevelGeneric {
    PriceType      prc{};
    SizeType       agg_sz{};
    NumOrdersType  num_orders{};
    FeedTimeType   feed_time{};
    UniqueUpdateID uuid{};
    void*          plist{nullptr};
};

using GeminiPrcLevelGeneric = AggBookPrcLevel;

// Size: Exactly 64 bytes (Fits 1 CPU Cache Line perfectly)
struct alignas(64) MapNode {
    uint64_t        key;
    GeminiPrcLevelGeneric value;
    uint32_t        left;
    uint32_t        right;
    uint32_t        parent;
    int32_t         height;
};

template <size_t MaxNodes>
class PreallocatedNodePool {
private:
    std::array<MapNode, MaxNodes + 1> storage_;
    uint32_t free_head_{1};

public:
    PreallocatedNodePool() noexcept {
        for (uint32_t i = 1; i < MaxNodes; ++i) {
            storage_[i].left = i + 1;
        }
        storage_[MaxNodes].left = NIL;
        storage_[NIL] = {0, {}, NIL, NIL, NIL, 0};
    }

    [[nodiscard]] [[gnu::always_inline]] inline uint32_t allocate(uint64_t key) noexcept {
        if (free_head_ == NIL) [[unlikely]] return NIL;
        uint32_t allocated_idx = free_head_;
        free_head_ = storage_[allocated_idx].left;

        MapNode& node = storage_[allocated_idx];
        node.key = key;
        node.left = NIL;
        node.right = NIL;
        node.parent = NIL;
        node.height = 1;
        return allocated_idx;
    }

    [[gnu::always_inline]] inline void deallocate(uint32_t idx) noexcept {
        if (idx == NIL) [[unlikely]] return;
        storage_[idx].left = free_head_;
        free_head_ = idx;
    }

    [[nodiscard]] [[gnu::always_inline]] inline const MapNode& get(uint32_t idx) const noexcept { return storage_[idx]; }
    [[nodiscard]] [[gnu::always_inline]] inline MapNode& get(uint32_t idx) noexcept { return storage_[idx]; }

    [[nodiscard]] [[gnu::always_inline]] inline const MapNode* get_ptr(uint32_t idx) const noexcept { return &storage_[idx]; }
    [[nodiscard]] [[gnu::always_inline]] inline MapNode* get_ptr(uint32_t idx) noexcept { return &storage_[idx]; }

    [[nodiscard]] [[gnu::always_inline]] inline uint32_t get_index(const MapNode* ptr) const noexcept {
        if (!ptr) return NIL;
        return static_cast<uint32_t>(ptr - storage_.data());
    }
};



// Forward declaration of Map container to satisfy Iterator bindings
template <size_t MaxNodes, typename Compare>
class FastNodeMap;

template <size_t MaxNodes, typename Compare>
class MapIterator {
private:
    const FastNodeMap<MaxNodes, Compare>* container_;
    MapNode*                              ptr_;

public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = MapNode;
    using difference_type   = std::ptrdiff_t;
    using pointer           = MapNode*;
    using reference         = MapNode&;

    MapIterator(const FastNodeMap<MaxNodes, Compare>* container, MapNode* ptr) noexcept
        : container_(container), ptr_(ptr) {}

    [[nodiscard]] inline reference operator*() const noexcept { return *ptr_; }
    [[nodiscard]] inline pointer   operator->() const noexcept { return ptr_; }

    inline MapIterator& operator++() noexcept {
        ptr_ = container_->next(ptr_);
        return *this;
    }

    inline MapIterator operator++(int) noexcept {
        MapIterator tmp = *this;
        ptr_ = container_->next(ptr_);
        return tmp;
    }

    [[nodiscard]] inline bool operator==(const MapIterator& o) const noexcept { return ptr_ == o.ptr_; }
    [[nodiscard]] inline bool operator!=(const MapIterator& o) const noexcept { return ptr_ != o.ptr_; }
};



template <size_t MaxNodes, typename Compare = std::less<uint64_t>>
class FastNodeMap {
private:
    PreallocatedNodePool<MaxNodes> pool_;
    uint32_t root_{NIL};
    size_t size_{0};
    [[no_unique_address]] Compare comp_;

    [[nodiscard]] [[gnu::always_inline]] inline int32_t get_height(uint32_t idx) const noexcept { return pool_.get(idx).height; }
    [[nodiscard]] [[gnu::always_inline]] inline int32_t get_balance(uint32_t idx) const noexcept {
        if (idx == NIL) return 0;
        return get_height(pool_.get(idx).left) - get_height(pool_.get(idx).right);
    }

    [[gnu::always_inline]] inline void update_height(uint32_t idx) noexcept {
        if (idx != NIL) {
            auto& node = pool_.get(idx);
            node.height = 1 + std::max(get_height(node.left), get_height(node.right));
        }
    }

    uint32_t rotate_right(uint32_t y) noexcept {
        uint32_t x = pool_.get(y).left;
        uint32_t T2 = pool_.get(x).right;
        pool_.get(x).right = y;
        pool_.get(y).left = T2;
        pool_.get(x).parent = pool_.get(y).parent;
        pool_.get(y).parent = x;
        if (T2 != NIL) pool_.get(T2).parent = y;
        update_height(y);
        update_height(x);
        return x;
    }

    uint32_t rotate_left(uint32_t x) noexcept {
        uint32_t y = pool_.get(x).right;
        uint32_t T2 = pool_.get(y).left;
        pool_.get(y).left = x;
        pool_.get(x).right = T2;
        pool_.get(y).parent = pool_.get(x).parent;
        pool_.get(x).parent = y;
        if (T2 != NIL) pool_.get(T2).parent = x;
        update_height(x);
        update_height(y);
        return y;
    }

    uint32_t rebalance(uint32_t idx) noexcept {
        update_height(idx);
        int32_t balance = get_balance(idx);
        if (balance > 1) {
            if (get_balance(pool_.get(idx).left) < 0) pool_.get(idx).left = rotate_left(pool_.get(idx).left);
            return rotate_right(idx);
        }
        if (balance < -1) {
            if (get_balance(pool_.get(idx).right) > 0) pool_.get(idx).right = rotate_right(pool_.get(idx).right);
            return rotate_left(idx);
        }
        return idx;
    }

    void balance_upward(uint32_t idx) noexcept {
        while (idx != NIL) {
            uint32_t p = pool_.get(idx).parent;
            uint32_t new_sub_root = rebalance(idx);
            if (p != NIL) {
                if (pool_.get(p).left == idx) pool_.get(p).left = new_sub_root;
                else pool_.get(p).right = new_sub_root;
                pool_.get(new_sub_root).parent = p;
            } else {
                root_ = new_sub_root;
                pool_.get(root_).parent = NIL;
            }
            idx = p;
        }
    }

    struct EmplaceResult { uint32_t new_root; bool inserted; MapNode* node_ptr; };
    template <typename... Args>
    EmplaceResult emplace_recursive(uint32_t node_idx, uint32_t parent_idx, uint64_t key, Args&&... args) noexcept {
        if (node_idx == NIL) {
            uint32_t fresh_idx = pool_.allocate(key);
            if (fresh_idx == NIL) [[unlikely]] return {NIL, false, nullptr};
            MapNode& fresh_node = pool_.get(fresh_idx);
            fresh_node.value = GeminiPrcLevelGeneric{std::forward<Args>(args)...};
            fresh_node.parent = parent_idx;
            ++size_;
            return {fresh_idx, true, &fresh_node};
        }
        auto& node = pool_.get(node_idx);
        EmplaceResult res;
        if (comp_(key, node.key)) {
            res = emplace_recursive(node.left, node_idx, key, std::forward<Args>(args)...);
            node.left = res.new_root;
        } else if (comp_(node.key, key)) {
            res = emplace_recursive(node.right, node_idx, key, std::forward<Args>(args)...);
            node.right = res.new_root;
        } else {
            return {node_idx, false, &node};
        }
        res.new_root = rebalance(node_idx);
        return res;
    }

    uint32_t get_extreme_node(uint32_t idx, bool find_min) const noexcept {
        uint32_t current = idx;
        if (find_min) {
            while (pool_.get(current).left != NIL) current = pool_.get(current).left;
        } else {
            while (pool_.get(current).right != NIL) current = pool_.get(current).right;
        }
        return current;
    }

public:
    using iterator       = MapIterator<MaxNodes, Compare>;
    using const_iterator = MapIterator<MaxNodes, Compare>;

    [[nodiscard]] inline iterator begin() noexcept { return iterator(this, first()); }
    [[nodiscard]] inline iterator end() noexcept   { return iterator(this, nullptr); }
    [[nodiscard]] inline const_iterator begin() const noexcept { return const_iterator(this, const_cast<FastNodeMap*>(this)->first()); }
    [[nodiscard]] inline const_iterator end() const noexcept   { return const_iterator(this, nullptr); }

    [[nodiscard]] [[gnu::always_inline]] inline size_t size() const noexcept { return size_; }
    [[nodiscard]] [[gnu::always_inline]] inline bool empty() const noexcept { return size_ == 0; }

    template <typename... Args>
    [[gnu::always_inline]] inline std::pair<MapNode*, bool> try_emplace(uint64_t key, Args&&... args) noexcept {
        auto [new_root, inserted, node_ptr] = emplace_recursive(root_, NIL, key, std::forward<Args>(args)...);
        root_ = new_root;
        if (root_ != NIL) pool_.get(root_).parent = NIL;
        return {node_ptr, inserted};
    }

    [[gnu::always_inline]] void erase(MapNode* node_ptr) noexcept {
        if (node_ptr == nullptr) [[unlikely]] return;
        uint32_t target_idx = pool_.get_index(node_ptr);
        if (target_idx == NIL) [[unlikely]] return;

        MapNode& node = pool_.get(target_idx);
        uint32_t balance_start_node = NIL;

        if (node.left == NIL || node.right == NIL) {
            uint32_t child = (node.left != NIL) ? node.left : node.right;
            uint32_t p = node.parent;
            if (child != NIL) pool_.get(child).parent = p;
            if (p != NIL) {
                if (pool_.get(p).left == target_idx) pool_.get(p).left = child;
                else pool_.get(p).right = child;
                balance_start_node = p;
            } else {
                root_ = child;
                if (root_ != NIL) pool_.get(root_).parent = NIL;
            }
            pool_.deallocate(target_idx);
            --size_;
        } else {
            uint32_t successor_idx = get_extreme_node(node.right, true);
            MapNode& successor = pool_.get(successor_idx);
            node.key = successor.key;
            node.value = successor.value;

            uint32_t child = successor.right;
            uint32_t p = successor.parent;
            if (child != NIL) pool_.get(child).parent = p;
            if (p != target_idx) {
                pool_.get(p).left = child;
                balance_start_node = p;
            } else {
                node.right = child;
                balance_start_node = target_idx;
            }
            pool_.deallocate(successor_idx);
            --size_;
        }
        if (balance_start_node != NIL) balance_upward(balance_start_node);
    }

    [[nodiscard]] [[gnu::always_inline]] inline const MapNode* find(uint64_t key) const noexcept {
        uint32_t current = root_;
        while (current != NIL) {
            const MapNode* node = pool_.get_ptr(current);
            if (key == node->key) [[likely]] return node;
            current = comp_(key, node->key) ? node->left : node->right;
        }
        return nullptr;
    }

    [[nodiscard]] [[gnu::always_inline]] inline MapNode* find(uint64_t key) noexcept {
        uint32_t current = root_;
        while (current != NIL) {
            MapNode* node = pool_.get_ptr(current);
            if (key == node->key) [[likely]] return node;
            current = comp_(key, node->key) ? node->left : node->right;
        }
        return nullptr;
    }

    [[nodiscard]] [[gnu::always_inline]] inline MapNode* first() noexcept {
        if (root_ == NIL) return nullptr;
        return pool_.get_ptr(get_extreme_node(root_, true));
    }

    [[nodiscard]] [[gnu::always_inline]] inline const MapNode* first() const noexcept {
        if (root_ == NIL) return nullptr;
        return pool_.get_ptr(get_extreme_node(root_, true));
    }

    [[nodiscard]] [[gnu::always_inline]] inline MapNode* next(const MapNode* current) const noexcept {
        if (!current) return nullptr;
        uint32_t idx = pool_.get_index(current);
        if (current->right != NIL) return const_cast<MapNode*>(pool_.get_ptr(get_extreme_node(current->right, true)));
        uint32_t p = current->parent;
        while (p != NIL && idx == pool_.get(p).right) {
            idx = p;
            p = pool_.get(p).parent;
        }
        return p == NIL ? nullptr : const_cast<MapNode*>(pool_.get_ptr(p));
    }

    [[gnu::always_inline]] inline size_t extract_volumes_simd(uint32_t* __restrict__ destination_array) const noexcept {
        uint32_t* aligned_dest = static_cast<uint32_t*>(__builtin_assume_aligned(destination_array, 64));
        size_t count = 0;
        MapNode* curr = const_cast<FastNodeMap*>(this)->first();
        while (curr != nullptr) {
            aligned_dest[count++] = curr->value.agg_sz;
            curr = next(curr);
        }
        return count;
    }
};

alignas(64) std::array<uint32_t, 1024> simd_scratch_buffer;

__attribute__((noinline)) uint64_t compute_aggregate_shares_simd(const uint32_t* __restrict__ data, size_t size) {
    const uint32_t* aligned_data = static_cast<const uint32_t*>(__builtin_assume_aligned(data, 64));
    uint64_t total_volume = 0;

    #pragma GCC ivdep
    for (size_t i = 0; i < size; ++i) {
        total_volume += aligned_data[i];
    }
    return total_volume;
}

#endif // BOOKBUILDERSPEC2_ITER_H_INCLUDED
