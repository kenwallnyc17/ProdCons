#ifndef GEMINI_NASD_ORDERBOOK_1_H_INCLUDED
#define GEMINI_NASD_ORDERBOOK_1_H_INCLUDED

#include <iostream>
#include <vector>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <cassert>

constexpr std::size_t CACHE_LINE_SIZE = 64;
constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

// --- TYPE OPTIMIZATIONS FOR MAXIMUM SPEED ---
using Price = std::uint32_t;       // Streamlined down to uint32_t per user constraint
using Quantity = std::uint32_t;    // Scaled down to uint32_t per user constraint
using OrderId = std::uint64_t;     // Stays 64-bit to prevent random hash truncation

enum class Side : std::uint8_t {
    BUY  = 0,
    SELL = 1
};

// Simulated specialized order list stub (to be implemented separately)
struct SpecializedOrderList {
    void* internal_ptr{nullptr};
};

struct alignas(CACHE_LINE_SIZE) OrderNode {
    OrderId order_id;
    Price price;
    Quantity quantity;
    std::uint64_t global_price_idx;
    Side side;

    std::uint32_t prev_order_idx{INVALID_INDEX};
    std::uint32_t next_order_idx{INVALID_INDEX};
};

struct alignas(CACHE_LINE_SIZE) PriceLevel {
    Quantity total_volume{0};
    std::uint32_t order_count{0};
    std::uint64_t feed_time{0}; // Metadata tracker per level
    std::uint64_t uuid{0};      // Metadata level key tracker

    SpecializedOrderList specialized_list; // Hook point for custom structure

    std::uint32_t head_order_idx{INVALID_INDEX};
    std::uint32_t tail_order_idx{INVALID_INDEX};
};

struct PricePage {
    static constexpr uint32_t PAGE_BITS = 12;
    static constexpr uint32_t PAGE_SIZE = 1 << PAGE_BITS;
    static constexpr uint32_t PAGE_MASK = PAGE_SIZE - 1;

    PriceLevel slots[PAGE_SIZE];
};

// --- CUSTOM ZERO-ALLOCATION MEMORY POOLS ---
class OrderNodePool {
private:
    std::vector<OrderNode> pool_;
    std::vector<std::uint32_t> free_stack_;

public:
    OrderNodePool(std::size_t capacity) {
        pool_.resize(capacity);
        free_stack_.reserve(capacity);
        for (std::size_t i = 0; i < capacity; ++i) {
            free_stack_.push_back(static_cast<std::uint32_t>((capacity - 1) - i));
        }
    }

    inline std::uint32_t allocate() {
        if (free_stack_.empty()) [[unlikely]] {
            throw std::bad_alloc(); // Core pool capacity exceeded
        }
        std::uint32_t idx = free_stack_.back();
        free_stack_.pop_back();
        return idx;
    }

    inline void deallocate(std::uint32_t idx) {
        pool_[idx] = OrderNode{}; // Reset memory footprint
        free_stack_.push_back(idx);
    }

    inline OrderNode& operator[](std::uint32_t idx) { return pool_[idx]; }
    inline const OrderNode& operator[](std::uint32_t idx) const { return pool_[idx]; }
};

class PageAllocator {
private:
    std::vector<std::unique_ptr<PricePage>> memory_chunks_;

public:
    inline PricePage* allocate_page() {
        memory_chunks_.push_back(std::make_unique<PricePage>());
        return memory_chunks_.back().get();
    }
    // Deallocation logic is naturally governed on engine shutdown contexts
};

class UnboundedOrderBook {
private:
    std::vector<PricePage*> price_directory_;
    PageAllocator page_pool_;
    OrderNodePool order_pool_;

    // Maps Random 64-bit Order ID to its current internal memory index location
    std::unordered_map<OrderId, std::uint32_t> random_id_map_;

  //  Price tick_size_;
    std::uint64_t level_uuid_counter_{1};

    const bool is_buy_book_;
    // Set to a massive upper bound (e.g., 500,000,000 for a max price of $5,000,000.00)
    const Price max_inverted_price_;
    const Price tick_size_;

   // inline std::uint64_t price_to_raw_index(Price price) const { return price / tick_size_; }
    inline std::uint64_t price_to_raw_index(Price price) const
    {
        if (is_buy_book_) [[likely]] {
            // High Buy Price -> Small Array Index (Front of the book)
            // Low Buy Price  -> Large Array Index (Deep back of the book)
            return (max_inverted_price_ - price) / tick_size_;
        } else {
            // Low Sell Price -> Small Array Index (Front of the book)
            // High Sell Price -> Large Array Index (Deep back of the book)
            return price / tick_size_;
        }
    }

    inline Price raw_index_to_price(std::uint64_t raw_idx) const
    {
        if (is_buy_book_) {
            // Reverse the math to get the original high buy price back
            return max_inverted_price_ - (raw_idx * tick_size_);
        } else {
            return raw_idx * tick_size_;
        }
    }

    inline PriceLevel& get_or_create_level(std::uint64_t raw_idx, std::uint64_t current_time) {
        std::uint64_t page_idx = raw_idx >> PricePage::PAGE_BITS;
        std::uint64_t offset = raw_idx & PricePage::PAGE_MASK;

        if (page_idx >= price_directory_.size()) [[unlikely]] {
            price_directory_.resize(page_idx + 1, nullptr);
        }
        if (!price_directory_[page_idx]) [[unlikely]] {
            price_directory_[page_idx] = page_pool_.allocate_page();
        }

        PriceLevel& level = price_directory_[page_idx]->slots[offset];
        if (level.uuid == 0) {
            level.uuid = level_uuid_counter_++;
            level.feed_time = current_time;
        }
        return level;
    }

    inline PriceLevel* get_level_ptr(std::uint64_t raw_idx) const {
        std::uint64_t page_idx = raw_idx >> PricePage::PAGE_BITS;
        std::uint64_t offset = raw_idx & PricePage::PAGE_MASK;
        if (page_idx >= price_directory_.size() || !price_directory_[page_idx]) return nullptr;
        return &price_directory_[page_idx]->slots[offset];
    }

public:
    UnboundedOrderBook(Price tick_size, bool is_buy, Price max_price_ceiling, std::size_t initial_pool_size)
        : order_pool_(initial_pool_size), is_buy_book_(is_buy), max_inverted_price_(max_price_ceiling)
        , tick_size_(tick_size) {
        random_id_map_.reserve(initial_pool_size);
    }

    // --- HOT PATH: O(1) ADD ORDER ---
    bool add_order(OrderId id, Price price, Quantity qty, Side side, std::uint64_t timestamp) {
        std::uint64_t raw_idx = price_to_raw_index(price);
        PriceLevel& level = get_or_create_level(raw_idx, timestamp);

        std::uint32_t node_idx = order_pool_.allocate();
        OrderNode& node = order_pool_[node_idx];
        node.order_id = id;
        node.price = price;
        node.quantity = qty;
        node.global_price_idx = raw_idx;
        node.side = side;

        random_id_map_[id] = node_idx;
        level.total_volume += qty;
        level.order_count++;
        level.feed_time = timestamp;

        // Link node to target price level list tail (FIFO Time-Priority Queue)
        if (level.tail_order_idx == INVALID_INDEX) {
            level.head_order_idx = node_idx;
            level.tail_order_idx = node_idx;
        } else {
            std::uint32_t old_tail_idx = level.tail_order_idx;
            order_pool_[old_tail_idx].next_order_idx = node_idx;
            node.prev_order_idx = old_tail_idx;
            level.tail_order_idx = node_idx;
        }
        return true;
    }

    // --- HOT PATH: O(1) PARTIAL CANCEL ---
    bool cancel_order(OrderId id, Quantity cancel_qty, std::uint64_t timestamp) {
        auto it = random_id_map_.find(id);
        if (it == random_id_map_.end()) return false;

        std::uint32_t node_idx = it->second;
        OrderNode& node = order_pool_[node_idx];

        PriceLevel* level_ptr = get_level_ptr(node.global_price_idx);
        if (!level_ptr) return false;

        // Safety cap: Cannot cancel more volume than remaining shares on node
        if (cancel_qty >= node.quantity) {
            return delete_order(id, timestamp);
        }

        node.quantity -= cancel_qty;
        level_ptr->total_volume -= cancel_qty;
        level_ptr->feed_time = timestamp;
        return true;
    }

    // --- HOT PATH: O(1) FULL ORDER DELETE ---
    bool delete_order(OrderId id, std::uint64_t timestamp) {
        auto it = random_id_map_.find(id);
        if (it == random_id_map_.end()) return false;

        std::uint32_t node_idx = it->second;
        const OrderNode& node = order_pool_[node_idx];

        PriceLevel* level_ptr = get_level_ptr(node.global_price_idx);
        if (level_ptr) {
            PriceLevel& level = *level_ptr;
            level.total_volume -= node.quantity;
            level.order_count--;
            level.feed_time = timestamp;

            if (node.prev_order_idx != INVALID_INDEX) {
                order_pool_[node.prev_order_idx].next_order_idx = node.next_order_idx;
            } else {
                level.head_order_idx = node.next_order_idx;
            }

            if (node.next_order_idx != INVALID_INDEX) {
                order_pool_[node.next_order_idx].prev_order_idx = node.prev_order_idx;
            } else {
                level.tail_order_idx = node.prev_order_idx;
            }
        }

        random_id_map_.erase(it);
        order_pool_.deallocate(node_idx);
        return true;
    }

    // --- HOT PATH: O(1) ATOMIC ORDER REPLACE (REPLACES INTERNAL LOOKUP IDS) ---
    bool replace_order(OrderId old_id, OrderId new_id, Price new_price, Quantity new_qty, std::uint64_t timestamp) {
        auto it = random_id_map_.find(old_id);
        if (it == random_id_map_.end()) [[unlikely]] return false;

        std::uint32_t node_idx = it->second;
        OrderNode& node = order_pool_[node_idx];

        std::uint64_t old_raw_idx = node.global_price_idx;
        std::uint64_t new_raw_idx = price_to_raw_index(new_price);
        Side order_side = node.side;

        // Case A: Price Migration (Time priority completely broken)
        if (old_raw_idx != new_raw_idx) {
            PriceLevel* old_level_ptr = get_level_ptr(old_raw_idx);
            if (old_level_ptr) {
                PriceLevel& old_level = *old_level_ptr;
                old_level.total_volume -= node.quantity;
                old_level.order_count--;
                old_level.feed_time = timestamp;

                if (node.prev_order_idx != INVALID_INDEX) {
                    order_pool_[node.prev_order_idx].next_order_idx = node.next_order_idx;
                } else {
                    old_level.head_order_idx = node.next_order_idx;
                }

                if (node.next_order_idx != INVALID_INDEX) {
                    order_pool_[node.next_order_idx].prev_order_idx = node.prev_order_idx;
                } else {
                    old_level.tail_order_idx = node.prev_order_idx;
                }
            }

            // Transfer node identities to new parameters
            node.order_id = new_id;
            node.price = new_price;
            node.quantity = new_qty;
            node.global_price_idx = new_raw_idx;

            // Re-hash table linkage for the identification update
            random_id_map_.erase(it);
            random_id_map_[new_id] = node_idx;

            PriceLevel& new_level = get_or_create_level(new_raw_idx, timestamp);
            new_level.total_volume += new_qty;
            new_level.order_count++;
            new_level.feed_time = timestamp;

            if (new_level.tail_order_idx == INVALID_INDEX)
            {
                new_level.head_order_idx = node_idx;
                new_level.tail_order_idx = node_idx;
                node.prev_order_idx = INVALID_INDEX;
                node.next_order_idx = INVALID_INDEX;
            }
            else {
                std::uint32_t old_tail_idx = new_level.tail_order_idx;
                order_pool_[old_tail_idx].next_order_idx = node_idx;node.prev_order_idx = old_tail_idx;
                node.next_order_idx = INVALID_INDEX;
                new_level.tail_order_idx = node_idx;
            }
            return true;
        }
        // Case B: Size Reduction (Time priority preserved)

        if (new_qty <= node.quantity)
        {
            PriceLevel* level_ptr = get_level_ptr(old_raw_idx);
            if (!level_ptr)
                return false;

            level_ptr->total_volume -= (node.quantity - new_qty);
            level_ptr->feed_time = timestamp;
            node.quantity = new_qty;
            node.order_id = new_id;
            random_id_map_.erase(it);
            random_id_map_[new_id] = node_idx;

            return true;
        }

        // Case C: Size Expansion (Time priority broken, dropped to tail of same level)
        PriceLevel* level_ptr = get_level_ptr(old_raw_idx);
        if (!level_ptr) return false;
        PriceLevel& level = *level_ptr;
        level.total_volume = (level.total_volume - node.quantity) + new_qty;
        level.feed_time = timestamp;
        node.quantity = new_qty;
        node.order_id = new_id;
        random_id_map_.erase(it);
        random_id_map_[new_id] = node_idx;
        if (level.tail_order_idx != node_idx)
        {
            if (node.prev_order_idx != INVALID_INDEX)
            {
                order_pool_[node.prev_order_idx].next_order_idx = node.next_order_idx;
        } else {
            level.head_order_idx = node.next_order_idx;
        }
        order_pool_[node.next_order_idx].prev_order_idx = node.prev_order_idx;
        std::uint32_t old_tail_idx = level.tail_order_idx;
        order_pool_[old_tail_idx].next_order_idx = node_idx;
        node.prev_order_idx = old_tail_idx;
        node.next_order_idx = INVALID_INDEX;
        level.tail_order_idx = node_idx;
        }
        return true;
    }

    // Diagnostic validation method for the test harness verification loop
    void inspect_level(Price price, uint32_t expected_vol, uint32_t expected_count)
    {
        PriceLevel* lvl = get_level_ptr(price_to_raw_index(price));
        if (!lvl) {std::cout << "[Verification] Price Level " << price << " has no allocations.\n";
            return;
        }
        std::cout << "[Level " << price << "] Vol=" << lvl->total_volume<< " (Expected: "
            << expected_vol << ") | Counts=" << lvl->order_count<< " (Expected: "
            << expected_count << ") | UUID=" << lvl->uuid << "\n";
        assert(lvl->total_volume == expected_vol);
        assert(lvl->order_count == expected_count);
    }

    };
    // ============================================================================// TEST HARNESS: PARSING DECODED NORMALIZED NASDAQ ITCH MESSAGES// ============================================================================

    enum class ItchMsgType : char {
        ADD_ORDER            = 'A', // Add Order - No MPID Attribution
        ORDER_EXECUTE        = 'E', // Order Executed Message (Full/Partial Fill)
        ORDER_CANCEL         = 'X', // Order Cancel Message (Partial Reduction)
        ORDER_DELETE         = 'D', // Order Delete Message (Full Purge)
        ORDER_REPLACE        = 'U'  // Order Replace Message (Price / Size Change)
    };
    struct NormalizedItchPacket
    {
        ItchMsgType type;
        std::uint64_t timestamp; // Nanoseconds since midnight
        OrderId order_id;        // 64-Bit Unique Exchange Ref ID
        Price price;             // Scaled 4-decimal integer representation
        Quantity quantity;       // Shares depth target
        Side side;               // Buy / Sell identification byte
        OrderId new_order_id;    // populated only during Type 'U' Replace events
    };

    int test_GEMINI_NASD_ordrBook_1()
    {
        //Price tick_size, bool is_buy, Price max_price_ceiling, std::size_t initial_pool_size
        // Initialize container tracking structure to accept up to 100,000 active nodes
        UnboundedOrderBook b_book(1, true, 2000000, 100000);
        std::cout << "--- Initializing Decoded NASDAQ ITCH Packet Injection Vector ---\n\n";
        // Mock stream representing realistic network packet sequence deliveries
        std::vector<NormalizedItchPacket> feed_stream =
        {
            // 1. Add baseline liquidity to Buy Side
            {ItchMsgType::ADD_ORDER, 34200000000000ULL, 5001, 10050, 1000, Side::BUY, 0 },
            {ItchMsgType::ADD_ORDER, 34200000001200ULL, 5002, 10050, 400,  Side::BUY, 0 },
            // 2. Partially reduce liquidity on order 5001 via execution or partial cancel
            { ItchMsgType::ORDER_CANCEL, 34200000250000ULL, 5001, 0, 300, Side::BUY, 0 },
            // 3. Perform atomic inline replacement update targeting old order 5002
            // Drops size from 400 down to 200 while maintaining same price (Case B - Priority Intact)
            { ItchMsgType::ORDER_REPLACE, 34200000500000ULL, 5002, 10050, 200, Side::BUY, 6001 },

            // 4. Wipe out remaining order balance of 5001 completely
            { ItchMsgType::ORDER_DELETE, 34200000990000ULL, 5001, 0, 0, Side::BUY, 0 },
            // 5. Relocate order 6001 to a higher price level premium step (Case A - Relocation Queue)
            { ItchMsgType::ORDER_REPLACE, 34200001200000ULL, 6001, 10060, 500, Side::BUY, 7001 }
        };

        // Fast iteration packet engine ingestion pipeline loop simulation

        for (const auto& packet : feed_stream)
        {
            switch (packet.type)
            {
                case ItchMsgType::ADD_ORDER:
                    b_book.add_order(packet.order_id, packet.price, packet.quantity, packet.side, packet.timestamp);
                    break;
                case ItchMsgType::ORDER_CANCEL:
                    b_book.cancel_order(packet.order_id, packet.quantity, packet.timestamp);
                    break;
                case ItchMsgType::ORDER_DELETE:
                    b_book.delete_order(packet.order_id, packet.timestamp);
                    break;
                case ItchMsgType::ORDER_REPLACE:
                    b_book.replace_order(packet.order_id, packet.new_order_id, packet.price, packet.quantity, packet.timestamp);
                    break;
                default:
                    break;
            }
        }

        // Run structural diagnostics on resulting price levels to verify accuracy

        b_book.inspect_level(10050, 0, 0);   // Both orders migrated or fully cleared out
        b_book.inspect_level(10060, 500, 1); // 7001 sits alone with volume 500

        {
            UnboundedOrderBook s_book(1, false, 2000000, 100000);
            std::vector<NormalizedItchPacket> feed_stream =
            {
                // 1. Add baseline liquidity to Buy Side
                {ItchMsgType::ADD_ORDER, 34200000000000ULL, 5001, 10050, 1000, Side::SELL, 0 },
                {ItchMsgType::ADD_ORDER, 34200000001200ULL, 5002, 10050, 400,  Side::SELL, 0 },
                // 2. Partially reduce liquidity on order 5001 via execution or partial cancel
                { ItchMsgType::ORDER_CANCEL, 34200000250000ULL, 5001, 0, 300, Side::SELL, 0 },
                // 3. Perform atomic inline replacement update targeting old order 5002
                // Drops size from 400 down to 200 while maintaining same price (Case B - Priority Intact)
                { ItchMsgType::ORDER_REPLACE, 34200000500000ULL, 5002, 10050, 200, Side::SELL, 6001 },

                // 4. Wipe out remaining order balance of 5001 completely
                { ItchMsgType::ORDER_DELETE, 34200000990000ULL, 5001, 0, 0, Side::SELL, 0 },
                // 5. Relocate order 6001 to a higher price level premium step (Case A - Relocation Queue)
                { ItchMsgType::ORDER_REPLACE, 34200001200000ULL, 6001, 10060, 500, Side::SELL, 7001 }
            };

            for (const auto& packet : feed_stream)
            {
                switch (packet.type)
                {
                    case ItchMsgType::ADD_ORDER:
                        s_book.add_order(packet.order_id, packet.price, packet.quantity, packet.side, packet.timestamp);
                        break;
                    case ItchMsgType::ORDER_CANCEL:
                        s_book.cancel_order(packet.order_id, packet.quantity, packet.timestamp);
                        break;
                    case ItchMsgType::ORDER_DELETE:
                        s_book.delete_order(packet.order_id, packet.timestamp);
                        break;
                    case ItchMsgType::ORDER_REPLACE:
                        s_book.replace_order(packet.order_id, packet.new_order_id, packet.price, packet.quantity, packet.timestamp);
                        break;
                    default:
                        break;
                }
            }

            // Run structural diagnostics on resulting price levels to verify accuracy

            s_book.inspect_level(10050, 0, 0);   // Both orders migrated or fully cleared out
            s_book.inspect_level(10060, 500, 1); // 7001 sits alone with volume 500
        }


        std::cout << "\n--- Ingestion Pipeline Diagnostic Confirmations Complete: All Checksums Match ---\n";

    return 0;
}

#endif // GEMINI_NASD_ORDERBOOK_1_H_INCLUDED
