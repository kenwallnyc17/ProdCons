struct alignas(64) MapNode {
    uint64_t        key;
    GeminiPrcLevelGeneric value;
    uint32_t        left;
    uint32_t        right;
    uint32_t        parent;
    int32_t         height;
};

// Chunk-size config: 65536 nodes per block keeps math to fast bit shifts (16 bits)
constexpr uint32_t CHUNK_SHIFT = 16;
constexpr uint32_t CHUNK_SIZE  = 1U << CHUNK_SHIFT;
constexpr uint32_t CHUNK_MASK  = CHUNK_SIZE - 1;

class DynamicNodePool {
private:
    std::vector<MapNode*> chunks_;
    uint32_t free_head_{NIL};
    uint32_t total_capacity_{0};

    // Allocates a brand new chunk and hooks it straight into the internal free-list chain
    void grow_pool() {
        // Allocate raw cache-aligned block memory via standard allocation handles
        MapNode* new_chunk = static_cast<MapNode*>(std::aligned_alloc(64, CHUNK_SIZE * sizeof(MapNode)));
        chunks_.push_back(new_chunk);

        uint32_t start_idx = total_capacity_;
        
        // Special case initialization for the absolute first chunk (index 0 is NIL)
        if (start_idx == 0) {
            new_chunk[0] = {0, {}, NIL, NIL, NIL, 0};
            for (uint32_t i = 1; i < CHUNK_SIZE - 1; ++i) {
                new_chunk[i].left = i + 1;
            }
            new_chunk[CHUNK_SIZE - 1].left = free_head_;
            free_head_ = 1;
        } else {
            // General initialization phase for subsequent blocks
            for (uint32_t i = 0; i < CHUNK_SIZE - 1; ++i) {
                new_chunk[i].left = start_idx + i + 1;
            }
            new_chunk[CHUNK_SIZE - 1].left = free_head_;
            free_head_ = start_idx;
        }
        total_capacity_ += CHUNK_SIZE;
    }

public:
    DynamicNodePool() {
        grow_pool(); // Seed the first memory block up-front
    }

    ~DynamicNodePool() noexcept {
        for (MapNode* chunk : chunks_) {
            std::free(chunk);
        }
    }

    // Explicit delete copy operations to prevent dual free faults
    DynamicNodePool(const DynamicNodePool&) = delete;
    DynamicNodePool& operator=(const DynamicNodePool&) = delete;

    [[nodiscard]] [[gnu::always_inline]] inline uint32_t allocate(uint64_t key) noexcept {
        if (free_head_ == NIL) [[unlikely]] {
            grow_pool(); // Automatically scale up capacity dynamically without invalidating links
        }
        
        uint32_t allocated_idx = free_head_;
        free_head_ = get(allocated_idx).left; 
        
        MapNode& node = get(allocated_idx);
        node.key = key;
        node.left = NIL;
        node.right = NIL;
        node.parent = NIL;
        node.height = 1;
        return allocated_idx;
    }

    [[gnu::always_inline]] inline void deallocate(uint32_t idx) noexcept {
        if (idx == NIL) [[unlikely]] return;
        get(idx).left = free_head_; 
        free_head_ = idx;
    }

    // O(1) Index lookup using high-speed division shifts and bitwise masking
    [[nodiscard]] [[gnu::always_inline]] inline const MapNode& get(uint32_t idx) const noexcept { 
        return chunks_[idx >> CHUNK_SHIFT][idx & CHUNK_MASK]; 
    }
    [[nodiscard]] [[gnu::always_inline]] inline MapNode& get(uint32_t idx) noexcept { 
        return chunks_[idx >> CHUNK_SHIFT][idx & CHUNK_MASK]; 
    }
    [[nodiscard]] [[gnu::always_inline]] inline const MapNode* get_ptr(uint32_t idx) const noexcept { 
        return &chunks_[idx >> CHUNK_SHIFT][idx & CHUNK_MASK]; 
    }
    [[nodiscard]] [[gnu::always_inline]] inline MapNode* get_ptr(uint32_t idx) noexcept { 
        return &chunks_[idx >> CHUNK_SHIFT][idx & CHUNK_MASK]; 
    }
    
    // Decodes a raw node pointer back to its virtual index coordinate path via linear scans
    [[nodiscard]] inline uint32_t get_index(const MapNode* ptr) const noexcept {
        if (!ptr) return NIL;
        for (size_t i = 0; i < chunks_.size(); ++i) {
            if (ptr >= chunks_[i] && ptr < chunks_[i] + CHUNK_SIZE) {
                return static_cast<uint32_t>((i << CHUNK_SHIFT) + (ptr - chunks_[i]));
            }
        }
        return NIL;
    }
};

#include <iterator>

template <typename Compare>
class FastNodeMap;

template <typename Compare>
class MapIterator {
private:
    const FastNodeMap<Compare>* container_;
    MapNode*                    ptr_;

public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = MapNode;
    using difference_type   = std::ptrdiff_t;
    using pointer           = MapNode*;
    using reference         = MapNode&;

    MapIterator(const FastNodeMap<Compare>* container, MapNode* ptr) noexcept 
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

#include <algorithm>
#include <utility>
#include <concepts>

template <typename Compare = std::less<uint64_t>>
class FastNodeMap {
private:
    DynamicNodePool pool_;
    uint32_t        root_{NIL};
    size_t          size_{0};
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
            MapNode& fresh_node = pool_.get(fresh_idx);
            fresh_node.value = PrcLevelGeneric{std::forward<Args>(args)...};
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
    using iterator       = MapIterator<Compare>;
    using const_iterator = MapIterator<Compare>;

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


// --- OLD FIXED-SIZE VARIABLES ---
// std::array<MapNode, MaxNodes + 1> storage_;
// uint32_t free_head_{1};

// --- NEW GROWING VARIABLES ---
std::vector<MapNode*> chunks_;
uint32_t free_head_{0}; // Initialized to 0 (NIL)
uint32_t total_capacity_{0};

// Sizing config: 65536 nodes per chunk handles math via 16-bit bitwise shifts
static constexpr uint32_t CHUNK_SHIFT = 16;
static constexpr uint32_t CHUNK_SIZE  = 1U << CHUNK_SHIFT;
static constexpr uint32_t CHUNK_MASK  = CHUNK_SIZE - 1;

// --- ADD THIS METHODS BLOCK TO YOUR POOL CLASS ---
void grow_pool() {
    MapNode* new_chunk = static_cast<MapNode*>(std::aligned_alloc(64, CHUNK_SIZE * sizeof(MapNode)));
    chunks_.push_back(new_chunk);

    uint32_t start_idx = total_capacity_;
    
    if (start_idx == 0) {
        // Safe placement new initialization for index 0 (The NIL sentinel node)
        ::new (static_cast<void*>(new_chunk)) MapNode{0, {}, NIL, NIL, NIL, 0};
        
        for (uint32_t i = 1; i < CHUNK_SIZE; ++i) {
            ::new (static_cast<void*>(&new_chunk[i])) MapNode{};
            new_chunk[i].left = (i < CHUNK_SIZE - 1) ? (i + 1) : free_head_;
        }
        free_head_ = 1;
    } else {
        // Initialize general sequential extensions
        for (uint32_t i = 0; i < CHUNK_SIZE; ++i) {
            ::new (static_cast<void*>(&new_chunk[i])) MapNode{};
            new_chunk[i].left = start_idx + i + 1;
        }
        new_chunk[CHUNK_SIZE - 1].left = free_head_;
        free_head_ = start_idx;
    }
    total_capacity_ += CHUNK_SIZE;
}

// Modify your constructor and destructor handles
PreallocatedNodePool() { grow_pool(); }
~PreallocatedNodePool() noexcept {
    for (MapNode* chunk : chunks_) {
        for (uint32_t i = 0; i < CHUNK_SIZE; ++i) chunk[i].~MapNode();
        std::free(chunk);
    }
}

// Update the allocation hook pass
[[nodiscard]] [[gnu::always_inline]] inline uint32_t allocate(uint64_t key) noexcept {
    if (free_head_ == NIL) [[unlikely]] grow_pool(); 
    
    uint32_t allocated_idx = free_head_;
    free_head_ = get(allocated_idx).left; 
    
    MapNode& node = get(allocated_idx);
    node.key = key; node.left = NIL; node.right = NIL; node.parent = NIL; node.height = 1;
    return allocated_idx;
}

// --- NEW O(1) BITWISE LOOKUPS ---
[[nodiscard]] [[gnu::always_inline]] inline const MapNode& get(uint32_t idx) const noexcept { return chunks_[idx >> CHUNK_SHIFT][idx & CHUNK_MASK]; }
[[nodiscard]] [[gnu::always_inline]] inline MapNode& get(uint32_t idx) noexcept { return chunks_[idx >> CHUNK_SHIFT][idx & CHUNK_MASK]; }
[[nodiscard]] [[gnu::always_inline]] inline const MapNode* get_ptr(uint32_t idx) const noexcept { return &chunks_[idx >> CHUNK_SHIFT][idx & CHUNK_MASK]; }
[[nodiscard]] [[gnu::always_inline]] inline MapNode* get_ptr(uint32_t idx) noexcept { return &chunks_[idx >> CHUNK_SHIFT][idx & CHUNK_MASK]; }

// --- REPLACED POINTER SUBTRACTION TRANSLATION ---
[[nodiscard]] inline uint32_t get_index(const MapNode* ptr) const noexcept {
    if (!ptr) return NIL;
    for (size_t i = 0; i < chunks_.size(); ++i) {
        if (ptr >= chunks_[i] && ptr < chunks_[i] + CHUNK_SIZE) {
            return static_cast<uint32_t>((i << CHUNK_SHIFT) + (ptr - chunks_[i]));
        }
    }
    return NIL;
}



