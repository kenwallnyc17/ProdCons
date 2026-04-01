#ifndef MEMORYCUSTOMIZATION_H_INCLUDED
#define MEMORYCUSTOMIZATION_H_INCLUDED

#include "Definitions.h"
//#include "MktDataSystemBaseDefinitions.h"
//#include "MktDataSystemLogging.h"

namespace CUSTOMMEMORY
{
//using namespace MKTDATASYSTEM::LOGGING;

/// assume single threaded access
/// ??? roundup blocksize to a provided alignment requirement (eg, 16)
template <uint64_t blocks_aligned_to = alignof(std::max_align_t)>
class MemoryPool
{
public:
    MemoryPool(const size_t blockSize, const size_t numBlocks, const size_t chunkGrowthFactor = 2) noexcept(false);
    ~MemoryPool();

    void* allocate() noexcept(false);
    void deallocate(void* ptr) noexcept;

private:
    struct FreeBlock {
        FreeBlock* next{};
    };

    char* poolStart_{};
    FreeBlock* freeList_{};
    const size_t blockSize_;
    const size_t numBlocks_;/// round up to fill a memory page
    size_t allocCnt_{};

    /// Add more chunks of memory dynamically when needed
    const size_t chunkGrowth_;
    struct Chunk
    {
        char* p{};
        size_t allocCnt{};
    };
    deque<Chunk> extraChunks_;
    // ... other members for managing the pool
};

template <uint64_t blocks_aligned_to>
MemoryPool<blocks_aligned_to>::MemoryPool(const size_t blockSize, const size_t numBlocks, const size_t chunkGrowthFactor) noexcept(false) :
                        blockSize_(roundupto.template operator()<blocks_aligned_to>(blockSize)),
                        numBlocks_(roundup_bytes.template operator()<huge_page_size_bytes>(numBlocks*blockSize_)/blockSize_),
                        chunkGrowth_(chunkGrowthFactor)
{
//    cout << "MemoryPool::MemoryPool(const size_t blockSize, const size_t numBlocks) "
//        << blockSize << ", " << numBlocks << endl;

    poolStart_ = new (std::align_val_t(huge_page_size_bytes)) char[blockSize_*numBlocks_]{};/// zero out
}

template <uint64_t blocks_aligned_to>
MemoryPool<blocks_aligned_to>::~MemoryPool()
{
    ::operator delete[](poolStart_, blockSize_*numBlocks_, std::align_val_t(huge_page_size_bytes));

    for(Chunk& el : extraChunks_)
    {
        ::operator delete[](el.p, blockSize_*numBlocks_/chunkGrowth_, std::align_val_t(huge_page_size_bytes));
    }
}

/// todo: align the returned pv -- done (the blocks are expected to be aligned according to the alignof(T))
template <uint64_t blocks_aligned_to>
void* MemoryPool<blocks_aligned_to>::allocate() noexcept(false)
{
    /// KMW : not sure which block is more likely
    if(freeList_ != nullptr)
    {
        void* pv{reinterpret_cast<void*>(freeList_)};
        freeList_ = freeList_->next;

        return pv;
    }
    else if(allocCnt_ < numBlocks_)
    {
        void* pv{reinterpret_cast<void*>(poolStart_ + allocCnt_*blockSize_)};
        ++allocCnt_;

        return pv;
    }
    else
    {
        if(extraChunks_.empty() || extraChunks_.back().allocCnt == numBlocks_/chunkGrowth_)
        {
            extraChunks_.emplace_back(new (std::align_val_t(huge_page_size_bytes)) char[blockSize_*numBlocks_/chunkGrowth_],0);
        }

        Chunk& cr{extraChunks_.back()};
        void* pv{reinterpret_cast<void*>(cr.p + cr.allocCnt*blockSize_)};
        ++cr.allocCnt;

        return pv;
    }

    /// issue error

    assert(false);//test

    return nullptr;
}

template <uint64_t blocks_aligned_to>
void MemoryPool<blocks_aligned_to>::deallocate(void* ptr) noexcept
{
    if(ptr != nullptr)
    {
        FreeBlock* pfb{reinterpret_cast<FreeBlock*>(ptr)};
        pfb->next = freeList_;
        freeList_ = pfb;
    }
}

///use in std::map
template <typename T>
class PoolAllocator
{
public:
    using value_type = T;

    PoolAllocator(MemoryPool<alignof(T)>& mempool) : s_pool(mempool)
    {
      //  SRLOG(Level::INFO) << "PoolAllocator::PoolAllocator(MemoryPool& mempool), sizeof(T), type(T) =  "
       //     << sizeof(T) << ", " << type_name<T>() << /*", " << numBlocks <<*/ endl;
    }

    template <typename U>
    PoolAllocator(const PoolAllocator<U>& ob) : s_pool(ob.s_pool)//sizeof(U), 1026)
    {
        cout << "PoolAllocator::PoolAllocator(const PoolAllocator<U>& ob), sizeof(U), type(U) = " << sizeof(U) << ", " << type_name<U>() << endl;
    }

    T* allocate(size_t n)
    {
        cout << "PoolAllocator::allocate n, sizeof(T), type(T) = " << n << ", " << sizeof(T) << ", " << type_name<T>() << endl;

        return reinterpret_cast<T*>(s_pool.allocate());
    }

    void deallocate(T* p, size_t n){cout << "PoolAllocator:: deallocate n = " << n << endl; s_pool.deallocate(reinterpret_cast<void*>(p));}

    // Required for stateless allocators (or allocators with shared state)
    bool operator==(const PoolAllocator& other) const { return false; }
    bool operator!=(const PoolAllocator& other) const { return true; }

    // ... other required methods like construct, destroy, max_size
private:
    template <typename U> friend struct PoolAllocator;/// allow all PoolAllocator<U>'s access to privates
    MemoryPool<alignof(T)>& s_pool; // Or a way to access a shared pool instance
};

}

#endif // MEMORYCUSTOMIZATION_H_INCLUDED
