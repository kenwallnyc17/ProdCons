#ifndef MKTDATASYSTEM_H_INCLUDED
#define MKTDATASYSTEM_H_INCLUDED

/// to do:
///     * add line arbitration type
///     * ?? where to add arbitration logic
///     * normalized add,rep,cancel,exec,status, ... msgs
///     * OrderObj
///     * aggregate book builder


/**

11/10/25
* XXX re-org code
* Tick's use more types rather than enums --> template specialize BB on side type, <other>
    * Wire contains enums/values
* (??) instead of ExchID, have Exch class that has ID enum, other exchange specific settings ??

* next: BookBuilder
    * XXX define agbookprclevel (prc,sz,num ords, UID, feed time)
    * XXX use above to define agbook tick and wire
    * XXX apply add, mod, etc to buy/sell book and determine if need to publish
    * XXX simple at first: return iter at start of publish level and # of levels down

* after: piece together Feed/Line handler to TickProcessor to publish
    * MDSystem has processing loop with dedicated processing core assigned
        loop: read packet -> packet::msg ->LineHandler -> TP ->Pub

    * define MyExchMsgs: ASCII packet with multi msgs and ASCII Line Handler

    * XXX add feed_time and UUID to Ticks and propagate to book levels
*/

/**
    > review Cancel logic for gap handling
    > XXX add feed time and uuid
    > requires logic on TickProcessor to make sure it handles the lineHandler type   eg, L3 msg LH needs L3 msg TP
    > published price should be double --> use price scale  (or, maybe publish the scale to the client)
    > XXX fastOID with dynamic sizes
    > where put fast OID and SS maps??
    > feed constants in single template param: exchid, feedid, <other> ???
**    > pooled mem allocate with modern C++  **
    > implement exec
    > add priority change flag from exchange
    > read real pcap
    > line arb
    > cross check
    > gap check
    > How will a binary be created?? compile everything into 1 binary at a time??
*/

/**
11/21/25 to do next:
    > pooled mem allocate with modern C++ -- for use in AggPrcLevel map and OrderID and SymbolState, maybe even within the internal vector object
        * properly align memory (use std::align )
        * XXX run out of memory: allow for dynamic expansion of the pool or heap allocation on the fly

    > XXX symbol lookup using symbol string cast to uint64_t : collisions not bad, insert heavy volume symbols first --> first in bucket
    > compare symbol lookup approaches (aside from provided index) - ptree,
    > XXX ?? inherit from map for booklevels and block move/cp, etc.
    > XXX add SSMgr (sic ??) to tickprocessor with allocator and fastSSHashMap settings

11/28/25: (updated 12/4/25)
    > finish pooled mem for fastHash maps: for allocating stored objects and for bucket vector
        * <see above>
        * XXX each LineHandler/TickProcessor has it's own instance of memory pool and aggbookallocator
        * bucket vector :
            * XXX erase: moves to end of vector??, want to deallocate the pointer (for mempool)
            * mempool for bucket vector
    > XXX add logger
    > clean up: remove old, add logging, break out into more files: BB, TickProcessor, SS
    > in logger:
        * use container that returns a pointer to the data rather than a copy
        * review and rename and modernize it

12/4/25 +
    > future:
        * read real pcap
        * efvi - see efsink code from solarflare
        * real exch decoders
        * shared memory pub lambda
1        * uuid -- need A+B line code
        * X stats
        * handle order list at price levels with priority change
1  X     * line arbitrage
1            > clean way to set line and PhysLine
        * feed time updated at prc level
        * X add constexpr and noexcept throughout
        * XXX alignment of MemoryPool class : align the returned pv  void* MemoryPool::allocate()
        * rdtscp impl. Windows???
1  XX      * L2 pub review: replace (??) multi level, mod vs replace
1        * book cleansing
1  X     * priority change flag -- only on Mod
        * NYSE Mod has side change possibility?? do other exchanges ?
        * ?? only allow exch's which have replace orders to compile the replace processing code
   X     * try flat_map with fixed size std::array for SockAddr -> V map instead of SockAddrHashMap
1       * how build different exch's -- make more user friendly
            * ARCA

*/

3/20/26
    logging keeps writing lines to the file --- problem with MtoNVariable_2025.h

#if 0

/// config loaded dynamically because some values are not static (cores, MC groups, ...)
/// Translate the provided config text file into a Config object and check that it's exch, feed, etc policies match
///     the MDSystem build types for LH
config file:

    ID's : exch, feed

    input : file/ multicast groups broken into groups

    publish: L3 ?, L2 (subscription, shd mem type, ROCE)

    processing cores: matching the number of input groups


#endif // 0

#include "MktDataSystemInputs.h"
#include "MktDataSystemSymbolStates.h"
#include "MktDataSystemTicks.h"
#include "MktDataSystemFeedHandler.h"
#include "MktDataSystemBB.h"
#include "MktDataSystemPub.h"

namespace MKTDATASYSTEM
{

template<typename D>
concept IsExchDecoder = requires (D d){d.f();};///(exchID && feedid && ); /// finish this

/// KMW : look into this requires clause
template<typename D>
concept IsL3ExchDecoder = IsExchDecoder<D> && requires (D d){d.process_add(/*PacketMsg&*/);};


struct Config
{

};

#if 0

using namespace MKTDATASYSTEM::DEFINITIONS;

/// I : input (file, efvi, ...)
/// FH : exch Feed and Line handler -- must provide exchID and FeedFormat
/// TP : tick processor -- what to do with ticks generated by LH
/// BB : book builder
///  L3 and L2 publisher types (TCP subscription, shared mem, ROCE,...)

/// if have L2PUBS, then they need a BB (Aggregate Book Builder)

template<typename I, ExchID XID, FeedFormat FID,
        template<typename,typename>typename FH,
        template<typename,typename>typename TP,
        template<typename,typename>typename BB, typename...PUBS>//, typename...L2PUBS>
//requires (((sizeof...(L3PUBS) > 0) ? IsL3Decoder<FH> : true) && IsExchDecoder<FH>)
struct MDSystem
{
    MDSystem(const Config&);
};


void test_mktsystem_1()
{
    cout << "\n test_mktsystem_1" << endl;
/*
    FastOID<2> fastoid;

    Itch5FeedHandler ifh;

    Itch5LineHandler* itchLH_p = ifh.createLH({new Line({new PhysLine, new PhysLine}), new Line({new PhysLine, new PhysLine})});

    {
        using AddMsgItch = MKTDATASYSTEM::TICK::Add<ExchID::NYSE,FeedFormat::ITCH5>;

        OrderID oid{12345};
        PriceType p{1};
        SizeType sz{2};
        SeqNumType msn{17};

        AddMsgItch ad("ken", oid, p, sz, Side::Buy, msn);

        cout << ad.msg_seq_num << ", " << ad.sym << endl;
    }
*/
    using namespace MKTDATASYSTEM::DEFINITIONS;
    using namespace MKTDATASYSTEM::BOOKBUILDER;
    using namespace MKTDATASYSTEM::TICK;
    using namespace MKTDATASYSTEM::SYMBOLSTATES;
    using namespace MKTDATASYSTEM::PUB;

    auto l2pub1 = [](const QuoteLevelsOneSide<ExchID::NASDAQ, FeedFormat::ITCH5>& bk_delta,
                     const SymbolState*){cout <<"qlos : " << bk_delta << endl;};

    ExchID exch_id{ExchID::NASDAQ};
    FeedFormat feed_id{FeedFormat::ITCH5};

    MKTDATASYSTEM::TICKPROCESSOR::L2TickProcessor<ExchID::NASDAQ, FeedFormat::ITCH5,
            BookBuilder,
            decltype(l2pub1)> l2tp;
#if 0
const /*SYMTYPE s*/MKTDATASYSTEM::DEFINITIONS::SymbolState* pss, const OrderID oid, const PriceType prc, const SizeType sz,
        const Side sd, const SeqNumType msn

        Rep(/*const char* s,*/ const OrderID ooid, const OrderID noid, const PriceType nprc,
        const SizeType nsz, const SeqNumType msn)
#endif
    const char* sym1 ="ken";
    SymIdxType sidx{17};

    SymbolObj so{sym1,sidx};
    decltype(l2pub1) L2lam1;

    FeedTimeType feed_time{176534};
    UniqueUpdateID uuid{1};

    {
        auto pss = GetSymbolState<MKTDATASYSTEM::SYMBOLSTATES::SymbolStateAggPriceBook>(sym1, sidx);
        Add<ExchID::NASDAQ, FeedFormat::ITCH5> add1(pss, 17, 68, 100, Side::Buy, feed_time++, uuid++);

        cout << __FILE__ << " : "<< __LINE__ <<endl;
        l2tp(add1);

        Add<ExchID::NASDAQ, FeedFormat::ITCH5> add2(pss, 18, 68, 100, Side::Buy, feed_time++, uuid++);
        l2tp(add2);

        Add<ExchID::NASDAQ, FeedFormat::ITCH5> add3(pss, 19, 69, 100, Side::Buy, feed_time++, uuid++);
        l2tp(add3);

        Del<ExchID::NASDAQ, FeedFormat::ITCH5> del1(18, feed_time++, uuid++);
        l2tp(del1);

        Del<ExchID::NASDAQ, FeedFormat::ITCH5> del2(19, feed_time++, uuid++);
        l2tp(del2);

        Can<ExchID::NASDAQ, FeedFormat::ITCH5> can1(17, 49, feed_time++, uuid++);
        l2tp(can1);

        Can<ExchID::NASDAQ, FeedFormat::ITCH5> can2(17, 51, feed_time++, uuid++);
        l2tp(can2);
    }

    {
        uint64_t strt_oid{20};
        auto pss = GetSymbolState<MKTDATASYSTEM::SYMBOLSTATES::SymbolStateAggPriceBook>(sym1, sidx);
        Add<ExchID::NASDAQ, FeedFormat::ITCH5> add1(pss, strt_oid, 68, 100, Side::Buy, feed_time++, uuid++);

        cout << __FILE__ << " : "<< __LINE__ <<endl;
        l2tp(add1);

        Add<ExchID::NASDAQ, FeedFormat::ITCH5> add2(pss, strt_oid + 1, 68, 100, Side::Buy, feed_time++, uuid++);
        l2tp(add2);

        Add<ExchID::NASDAQ, FeedFormat::ITCH5> add3(pss, strt_oid + 2, 69, 100, Side::Buy, feed_time++, uuid++);
        l2tp(add3);

        Rep<ExchID::NASDAQ, FeedFormat::ITCH5> rep1(strt_oid, strt_oid +3, DefaultPrice, 200, feed_time++, uuid++);
        l2tp(rep1);

        Rep<ExchID::NASDAQ, FeedFormat::ITCH5> rep2(strt_oid + 3, strt_oid +3, 67, DefaultSize, feed_time++, uuid++);
        l2tp(rep2);

        Rep<ExchID::NASDAQ, FeedFormat::ITCH5> rep3(strt_oid + 3, strt_oid +4, DefaultPrice, 700, feed_time++, uuid++);
        l2tp(rep3);

    }

    cout << "\n end test_mktsystem_1" << endl;
}

#endif
}

#endif // MKTDATASYSTEM_H_INCLUDED
