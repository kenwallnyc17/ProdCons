#ifndef MKTDATASYSTEMSYMBOLSTATES_H_INCLUDED
#define MKTDATASYSTEMSYMBOLSTATES_H_INCLUDED

#include "MktDataSystemBaseDefinitions.h"
#include "MktDataSystemContainers.h"
#include "MemoryCustomization.h"

namespace MKTDATASYSTEM::SYMBOLSTATES
{
using namespace MKTDATASYSTEM::DEFINITIONS;

struct Order;

struct AggBookPrcLevel : PrcLevelGeneric
{
    Order* pords{};/// linked list of Orders at this prc level -- time/priority sequenced

    AggBookPrcLevel(const PriceType prc, const SizeType agg_sz, const NumOrdersType num_orders,
                    const FeedTimeType feed_time, const UniqueUpdateID uuid, Order* pords)
                    : PrcLevelGeneric(prc, agg_sz, num_orders, feed_time, uuid), pords(pords) {}
};


using AggBookIter = map<PriceType, AggBookPrcLevel>::iterator;
using AggBookValType = map<PriceType, AggBookPrcLevel>::value_type;

using AggBookAlloctr = CUSTOMMEMORY::PoolAllocator<std::pair<const PriceType, AggBookPrcLevel>>;

/// what is used internally by the GCC impl -- what the pooled allocator is actually using
using AggBookAlloctrGCCNodeType = std::_Rb_tree_node<std::pair<const PriceType, AggBookPrcLevel> >;


/// block copy and move ops for the agg price books
template<typename Compare>
struct AggPrcLvlBook : map<PriceType, AggBookPrcLevel, Compare, AggBookAlloctr>
{
    using Base = map<PriceType, AggBookPrcLevel, Compare, AggBookAlloctr>;

    AggPrcLvlBook(AggBookAlloctr& alloc) : Base(alloc){}

    ///block cp and move
    AggPrcLvlBook(const AggPrcLvlBook&) = delete;
    AggPrcLvlBook& operator=(const AggPrcLvlBook&) = delete;
};


using BuyAggBook = AggPrcLvlBook<greater<PriceType>>;
using SellAggBook = AggPrcLvlBook<less<PriceType>>;

/// per active symbol
struct SymbolStateAggPriceBook : SymbolState
{
    SymbolStateAggPriceBook(const SymbolObj& so, AggBookAlloctr& aggbkAlloc) :
        SymbolState(so), buy_book_(aggbkAlloc), sell_book_(aggbkAlloc) {}

//private:
    BuyAggBook buy_book_;
    SellAggBook sell_book_;
    /// curr trading status : need SymbolStatus class/enum
};

struct Order
{
    OrderID oid;
    PriceType prc;/// better to use integer
    SizeType sz;
    Side    sd;
    AggBookIter ab_iter;/// iter to prc lvl in agg book
    SymbolState* pss;/// Add order sets this so that subsequent order msgs (mod/del/..) just have to look up the OrderObj

    Order(const OrderID oid, const PriceType prc, const SizeType sz, const Side sd, const AggBookIter ab_iter,
          SymbolState* pss) : oid(oid), prc(prc), sz(sz), sd(sd), ab_iter(ab_iter), pss(pss) {}

    OrderID id() const {return oid;}
};

/// one per TickProcessor
template<Constants C, derived_from<SymbolState> SS>
struct SymbolStateMgr
{
    using FastSymStateMap = MKTDATASYSTEM::CONTAINERS::FastUIntHashMap<SymIdxType, SS*>;

    SymbolStateMgr() : fssC(C.fastSymStateMapInputs),
        aggbk_mempool(sizeof(AggBookAlloctrGCCNodeType), C.symStatePrcLvl.Init, C.symStatePrcLvl.GrowthFactor),
        aggbkAlloc_(aggbk_mempool) {}

  //  SS* GetSymbolState(const char* ps, SymIdxType idx = DefaultSymIdx);
    SS* GetSymbolState(const SymbolObj& symob);

private:
    FastSymStateMap fssC;

    CUSTOMMEMORY::MemoryPool<alignof(AggBookAlloctrGCCNodeType)> aggbk_mempool;
    AggBookAlloctr aggbkAlloc_;//(mempool);
};

///KMW : try using the first 4 bytes of the symbol as the symidx if idx is not provided
#if 0
template<Constants C, derived_from<SymbolState> SS>
SS* SymbolStateMgr<C,SS>::GetSymbolState(const char* ps, SymIdxType idx)
{
   // static_assert(sizeof(SymIdxType) == 16);/// because using first 2 chars of symbol as default index
    SymbolState* pbss{};

    if(idx == DefaultSymIdx)/// sym idx not provided by exch
    {
        if( ps == nullptr)
            return nullptr;
//cout << __FILE__ << " : "<< __LINE__ << " : " << ps << " : " << idx <<endl;
        idx = *reinterpret_cast<SymIdxType*>(const_cast<char*>(ps));
//uint32_t idx16 = *reinterpret_cast<uint32_t*>(const_cast<char*>(ps));

cout << __FILE__ << " : "<< __LINE__ << " : " << ps << " : " << idx <<endl;
    }

    /// have idx populated at this point
    pbss = fssC.try_emplace(idx, ps, idx, aggbkAlloc_);

    return static_cast<SS*>(pbss);
}
#endif

template<Constants C, derived_from<SymbolState> SS>
SS* SymbolStateMgr<C,SS>::GetSymbolState(const SymbolObj& symob)
{
    SymbolState* pbss{};
    if(symob.idx == DefaultSymIdx)/// sym idx not provided by exch
    {
        if( symob.sym[0] == '\0')
            return nullptr;

        const_cast<SymIdxType&>(symob.idx) = *reinterpret_cast<SymIdxType*>(const_cast<char*>(&symob.sym[0]));
        SRLOG(Level::INFO) << "symob.idx = " <<symob.idx;
    }
    SRLOG(Level::INFO) << "symob.idx = " <<symob.idx;

    /// have idx populated at this point
    pbss = fssC.try_emplace(symob.idx, symob, aggbkAlloc_);

    return static_cast<SS*>(pbss);
}

}

#endif // MKTDATASYSTEMSYMBOLSTATES_H_INCLUDED
