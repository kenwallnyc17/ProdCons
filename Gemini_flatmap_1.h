#ifndef GEMINI_FLATMAP_1_H_INCLUDED
#define GEMINI_FLATMAP_1_H_INCLUDED

#include <immintrin.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

// --- CACHE-LINE ORDER BLOCK (EXACTLY 64 BYTES) ---
struct alignas(64) CondensedOrderBlock {
    struct CondensedOrder {
        uint64_t oid;
        uint32_t sz;
    };
    static_assert(sizeof(CondensedOrder) == 16);

    CondensedOrder block;            // Fixed explicit array of 3 elements (48 bytes)
    CondensedOrderBlock* next = nullptr;   // 8 bytes (64-bit system pointer)
    uint8_t padding = 0;             // 8 bytes padding -> 48 + 8 + 8 = 64 bytes total
};
static_assert(sizeof(CondensedOrderBlock) == 64);

// --- SPECIFIED PRICE LEVEL METADATA STRUCTURE (EXACTLY 40 BYTES) ---
struct PriceLevel {
    uint32_t prc;
    uint32_t aggregate_sz;
    uint16_t num_orders;
    uint64_t feed_time;
    uint64_t uuid;
    CondensedOrderBlock *pords;
};

// --- HIGH-SPEED VECTOR-CLEARED RESOURCE REUSE POOL ---
class UnifiedMemoryPool {
private:
    CondensedOrderBlock* m_order_pool = nullptr;
    CondensedOrderBlock* m_order_free_head = nullptr;
public:
    void init(size_t order_capacity) noexcept {
        void* order_mem = std::aligned_alloc(64, order_capacity * sizeof(CondensedOrderBlock));
        if (!order_mem) std::exit(EXIT_FAILURE);
        std::memset(order_mem, 0, order_capacity * sizeof(CondensedOrderBlock));
        m_order_pool = reinterpret_cast<CondensedOrderBlock*>(order_mem);
        for (size_t i = 0; i < order_capacity - 1; ++i) m_order_pool[i].next = &m_order_pool[i + 1];
        m_order_free_head = m_order_pool;
    }
    ~UnifiedMemoryPool() { std::free(m_order_pool); }

    inline CondensedOrderBlock* acquire_order_block() noexcept {
        CondensedOrderBlock* node = m_order_free_head;
        if (!node) [[unlikely]] return nullptr;
        m_order_free_head = node->next;
        node->next = nullptr;
        CondensedOrderBlock* aligned_node = std::assume_aligned<64>(node);

        // Zero first 32 bytes via 256-bit register blit
        _mm256_store_si256(reinterpret_cast<__m256i*>(aligned_node->block), _mm256_setzero_si256());
        // Zero remaining 16 bytes via 128-bit register blit
        _mm_store_si128(reinterpret_cast<__m128i*>(&aligned_node->block), _mm_setzero_si128());
        return aligned_node;
    }

    inline void release_order_block(CondensedOrderBlock* node) noexcept {
        if (!node) return;
        node->next = m_order_free_head;
        m_order_free_head = node;
    }
};
inline thread_local UnifiedMemoryPool g_hft_memory_pool;

#include <vector>
#include <utility>
#include <iostream>

template <bool IsBid>
class ChunkedPriceBook {
private:
    static constexpr uint32_t INITIAL_CAPACITY = 32;

    uint32_t* m_prices = nullptr;
    PriceLevel* m_levels = nullptr;
    uint32_t m_count = 0;
    uint32_t m_capacity = 0;

    void grow() noexcept {
        uint32_t new_cap = (m_capacity == 0) ? INITIAL_CAPACITY : m_capacity * 2;
        uint32_t padded_price_cap = (new_cap + 7) & ~7U; // Pad for safe 32-byte register reads

        uint32_t* new_prices = reinterpret_cast<uint32_t*>(std::aligned_alloc(32, padded_price_cap * sizeof(uint32_t)));
        PriceLevel* new_levels = reinterpret_cast<PriceLevel*>(std::aligned_alloc(64, new_cap * sizeof(PriceLevel)));

        std::memset(new_prices, 0, padded_price_cap * sizeof(uint32_t));
        std::memset(new_levels, 0, new_cap * sizeof(PriceLevel));

        if (m_prices) {
            std::memcpy(new_prices, m_prices, m_count * sizeof(uint32_t));
            std::memcpy(new_levels, m_levels, m_count * sizeof(PriceLevel));
            std::free(m_prices);
            std::free(m_levels);
        }

        m_prices = std::assume_aligned<32>(new_prices);
        m_levels = std::assume_aligned<64>(new_levels);
        m_capacity = new_cap;
    }

public:
    static constexpr uint32_t INVALID_INDEX = 0xFFFFFFFF;

    ChunkedPriceBook() { grow(); }
    ~ChunkedPriceBook() {
        for (uint32_t i = 0; i < m_count; ++i) {
            CondensedOrderBlock* current = m_levels[i].pords;
            while (current) {
                CondensedOrderBlock* next_node = current->next;
                g_hft_memory_pool.release_order_block(current);
                current = next_node;
            }
        }
        std::free(m_prices);
        std::free(m_levels);
    }

    inline PriceLevel* at(uint32_t index) noexcept {
        if (index >= m_count) [[unlikely]] return nullptr;
        return &m_levels[index];
    }

    inline const PriceLevel* at(uint32_t index) const noexcept {
        if (index >= m_count) [[unlikely]] return nullptr;
        return &m_levels[index];
    }

    inline uint32_t find_insert_index(uint32_t price, bool& out_exists) const noexcept {
        __m256i target = _mm256_set1_epi32(static_cast<int>(price));

        for (uint32_t i = 0; i < m_count; i += 8) {
            __m256i current_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(m_prices + i));

            int eq_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi32(target, current_vec));
            if (eq_mask > 0) {
                out_exists = true;
                return i + (__builtin_ctz(eq_mask) >> 2);
            }

            __m256i gt_cmp = IsBid ? _mm256_cmpgt_epi32(target, current_vec)
                                   : _mm256_cmpgt_epi32(current_vec, target);
            int gt_mask = _mm256_movemask_epi8(gt_cmp);
            if (gt_mask > 0) {
                out_exists = false;
                return i + (__builtin_ctz(gt_mask) >> 2);
            }
        }
        out_exists = false;
        return m_count;
    }

    inline uint32_t find(uint32_t price) const noexcept {
        bool exists = false;
        uint32_t idx = find_insert_index(price, exists);
        return exists ? idx : INVALID_INDEX;
    }

    std::pair<uint32_t, bool> try_emplace(uint32_t price, uint32_t sz, uint16_t num_orders, uint64_t feed_time, uint64_t uuid) noexcept {
        bool exists = false;
        uint32_t idx = find_insert_index(price, exists);
        if (exists) return { idx, false };

        if (m_count >= m_capacity) [[unlikely]] grow();

        std::memmove(m_prices + idx + 1, m_prices + idx, (m_count - idx) * sizeof(uint32_t));
        std::memmove(m_levels + idx + 1, m_levels + idx, (m_count - idx) * sizeof(PriceLevel));

        m_prices[idx] = price;
        m_levels[idx] = { price, sz, num_orders, feed_time, uuid, nullptr };
        m_count++;

        return { idx, true };
    }

    bool insert_order(uint32_t price, uint64_t oid, uint32_t sz, uint64_t timestamp, uint64_t uuid) noexcept {
        auto [flat_idx, is_new] = try_emplace(price, 0, 0, timestamp, uuid);
        return append_to_level(m_levels[flat_idx], oid, sz);
    }

    bool amend_order(uint32_t price, uint64_t oid, uint32_t new_sz) noexcept {
        uint32_t target_idx = find(price);
        if (target_idx == INVALID_INDEX) return false;
        PriceLevel& level = m_levels[target_idx];
        if (new_sz == 0) return remove_from_level(level, oid);

        __m256i target = _mm256_set1_epi64x(static_cast<int64_t>(oid));
        CondensedOrderBlock* current = level.pords;
        uint32_t depth = 0;

        while (current) {
            __m256i oids = _mm256_set_epi64x(
                                                0,                                // Lane 3: Zero Padding
                                            current->block[2].oid,            // Lane 2: Order 2 ID
                                            current->block[1].oid,            // Lane 1: Order 1 ID
                                            current->block[0].oid             // Lane 0: Order 0 ID
                                        );
            if (_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(target, oids))) > 0 || true) {
                for (uint32_t i = 0; i < 3; ++i) {
                    if (current->block[i].oid == oid && (depth * 3) + i < level.num_orders) {
                        level.aggregate_sz = (level.aggregate_sz - current->block[i].sz) + new_sz;
                        current->block[i].sz = new_sz;
                        return true;
                    }
                }
            }
            current = current->next; depth++;
        }
        return false;
    }

    bool erase(uint32_t index) noexcept {
        if (index >= m_count) [[unlikely]] return false;
        CondensedOrderBlock* current = m_levels[index].pords;
        while (current) {
            CondensedOrderBlock* next_node = current->next;
            g_hft_memory_pool.release_order_block(current);
            current = next_node;
        }

        std::memmove(m_prices + index, m_prices + index + 1, (m_count - index - 1) * sizeof(uint32_t));
        std::memmove(m_levels + index, m_levels + index + 1, (m_count - index - 1) * sizeof(PriceLevel));

        m_count--;
        m_prices[m_count] = 0;
        std::memset(&m_levels[m_count], 0, sizeof(PriceLevel));
        return true;
    }

    bool erase_by_price(uint32_t price) noexcept {
        uint32_t flat_idx = find(price);
        if (flat_idx == INVALID_INDEX) return false;
        return erase(flat_idx);
    }

    void match_against_book(uint32_t aggressive_sz) noexcept {
        while (aggressive_sz > 0 && m_count > 0) {
            PriceLevel& level = m_levels[0];
            if (level.aggregate_sz == 0) [[unlikely]] { erase(0); continue; }

            auto& top_order = std::assume_aligned<64>(level.pords)->block;
            if (aggressive_sz >= top_order.sz) {
                aggressive_sz -= top_order.sz;
                level.aggregate_sz -= top_order.sz;
                remove_from_level(level, top_order.oid);
                if (level.aggregate_sz == 0) erase(0);
            } else {
                top_order.sz -= aggressive_sz;
                level.aggregate_sz -= aggressive_sz;
                aggressive_sz = 0;
            }
        }
    }

    void print_top_of_book() const noexcept {
        std::cout << (IsBid ? "======== FLAT POOLED BID DEPTH ========\n" : "======== FLAT POOLED ASK DEPTH ========\n");
        for (uint32_t i = 0; i < std::min(m_count, 3U); ++i) {
            std::cout << "  Index: " << i << " | Price: " << m_prices[i] << " | Vol: " << m_levels[i].aggregate_sz << "\n";
        }
        std::cout << "=======================================\n\n";
    }

    inline uint32_t size() const noexcept { return m_count; }

private:
    bool append_to_level(PriceLevel& level, uint64_t oid, uint32_t sz) noexcept {
        if (!level.pords) {
            level.pords = g_hft_memory_pool.acquire_order_block();
            if (!level.pords) return false;
        }
        CondensedOrderBlock* current = level.pords;
        while (current->next) current = current->next;
        uint32_t inner_idx = level.num_orders % 3;
        if (level.num_orders > 0 && inner_idx == 0) {
            current->next = g_hft_memory_pool.acquire_order_block();
            current = current->next;
        }
        current->block[inner_idx] = {oid, sz};
        level.num_orders++;
        level.aggregate_sz += sz;
        return true;
    }

    bool remove_from_level(PriceLevel& level, uint64_t oid) noexcept {
        __m256i target = _mm256_set1_epi64x(static_cast<int64_t>(oid));
        CondensedOrderBlock* current = level.pords;
        CondensedOrderBlock* target_block = nullptr;
        uint32_t depth = 0, inner_idx = 0;
        bool found = false;

        while (current) {
            __m256i oids = _mm256_set_epi64x(
                                                0,                                // Lane 3: Zero Padding
                                            current->block[2].oid,            // Lane 2: Order 2 ID
                                            current->block[1].oid,            // Lane 1: Order 1 ID
                                            current->block[0].oid             // Lane 0: Order 0 ID
                                        );
            int mask = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(target, oids)));
            if (mask > 0 || true) {
                for (uint32_t i = 0; i < 3; ++i) {
                    if (current->block[i].oid == oid && (depth * 3) + i < level.num_orders) {
                        inner_idx = i; target_block = current; found = true; break;
                    }
                }
                if (found) break;
            }
            current = current->next; depth++;
        }
        if (!found) return false;

        uint32_t flat_idx = (depth * 3) + inner_idx;
        CondensedOrderBlock* current_node = target_block;
        uint32_t current_inner = inner_idx;

        for (uint32_t i = flat_idx; i < (uint32_t)level.num_orders - 1; ++i) {
            uint32_t next_inner = current_inner + 1;
            CondensedOrderBlock* next_node = current_node;
            if (next_inner == 3) { next_inner = 0; next_node = current_node->next; }
            current_node->block[current_inner] = next_node->block[next_inner];
            current_inner = next_inner; current_node = next_node;
        }
        current_node->block[current_inner] = {0, 0};
        level.num_orders--;

        if (level.num_orders == 0 && level.pords)
        {
            g_hft_memory_pool.release_order_block(level.pords);
            level.pords = nullptr;
        }
        return true;
    }
};

#include <iostream>

struct OrderBook {
    ChunkedPriceBook<true>  bids;
    ChunkedPriceBook<false> asks;
};

int main() {
    // 1. Pre-allocate pool arena context for 50,000 active order blocks
    g_hft_memory_pool.init(50000);
    OrderBook book;

    // 2. Validate associative try_emplace logic
    auto [idx1, is_new1] = book.bids.try_emplace(45000, 1000, 4, 1718717165, 9901);
    std::cout << "try_emplace 1 (Price 45000) -> New Level: " << std::boolalpha << is_new1
              << " | Index: " << idx1 << "\n";

    auto [idx2, is_new2] = book.bids.try_emplace(45000, 5000, 9, 1718717188, 9955);
    std::cout << "try_emplace 2 (Price 45000) -> New Level: " << std::boolalpha << is_new2
              << " | Index: " << idx2 << "\n\n";

    // 3. Load standard chronological order entries
    book.bids.insert_order(44980, 1002, 500, 1718717160, 9902);
    book.bids.insert_order(44950, 1003, 900, 1718717170, 9903);
    book.asks.insert_order(45020, 2001, 400, 1718717162, 8801);

    std::cout << "--- PRISTINE CONSOLIDATED MARKET DEPTH ---\n";
    book.bids.print_top_of_book();
    book.asks.print_top_of_book();

    // 4. Validate low-latency O(1) coordinate reference accessor updates
    uint32_t find_idx = book.bids.find(44980);
    if (find_idx != ChunkedPriceBook<true>::INVALID_INDEX) {
        PriceLevel* ptr = book.bids.at(find_idx);
        ptr->aggregate_sz += 300;
        std::cout << "Amended volume directly at Index " << find_idx << ". New Vol: " << book.bids.at(find_idx)->aggregate_sz << "\n\n";
    }

    // 5. Execute aggressive matching transaction sweeping 1000 shares from Bids
    std::cout << "INCOMING MARKET SELL ORDER: Sweeping 1000 shares from Bids...\n\n";
    book.bids.match_against_book(1000);

    std::cout << "--- ENGINE STATE POST-SWEEP ---\n";
    book.bids.print_top_of_book();

    return 0;
}

#endif // GEMINI_FLATMAP_1_H_INCLUDED
