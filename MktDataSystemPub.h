#ifndef MKTDATASYSTEMPUB_H_INCLUDED
#define MKTDATASYSTEMPUB_H_INCLUDED

//#include "MktDataSystemBaseDefinitions.h"
#include "MktDataSystemTicks.h"
#include "MktDataSystemSymbolStates.h"

namespace MKTDATASYSTEM::PUB
{
using namespace MKTDATASYSTEM::DEFINITIONS;
using namespace MKTDATASYSTEM::TICK;
using namespace MKTDATASYSTEM::SYMBOLSTATES;

ostream& operator<<(ostream& os, const PrcLevelGeneric& pl)
{
    os << pl.prc << " : " << pl.agg_sz << " : " << pl.num_orders << " : " << pl.feed_time << " : " << pl.uuid << endl;

    return os;
}

template<ExchID XID, FeedFormat FID>
ostream& operator<<(ostream& os, const QuoteLevelsOneSide<XID,FID>& qlos)
{
    os << "sym [" << qlos.sym << "] : seq_num [" << qlos.seq_num
        << "] : exch [" << int(to_underlying(qlos.exch_id))
        << "] : feed [ " << int(to_underlying(qlos.feed_id))
        << "] : msg type [" << int(to_underlying(qlos.msg_id))
        << "] : feed time [" << qlos.feed_time
        << "] : uuid [" << qlos.uuid
        << "]" << endl;
    os << "side [ " << int(to_underlying(qlos.sd))
        << " ] start lvl [" << int(qlos.start_lvl)
        << " ] num lvls [" << int(qlos.num)
        << "]" << endl;
    os << "prc : agg_sz : num_orders : feed_time : uuid" << endl;

    for(int i = 0; i < qlos.num; ++i)
    {
        os << qlos.lvls[i];
    }

    os << endl;

    return os;
}

ostream& operator<<(ostream& os, const SymbolStateAggPriceBook& ssagbk)
{
    auto pub_side = [&os](auto& bk)
    {
        auto it = bk.begin();
        for(uint8_t i = 0; i < MAX_PUB_LEVELS_TMP && it != bk.end(); ++i, ++it)
        {
            os << it->second;
        }

        os << endl;
    };

    os << "sym [" << ssagbk.symbol() << "] : sym idx [" << ssagbk.id()
        << "] : prc scale [" << ssagbk.prc_scale()
        << "]" << endl;

    os << "side [Buy] :" << endl;
    os << "prc : agg_sz : num_orders : feed_time : uuid" << endl;

    pub_side(ssagbk.buy_book_);

    os << "side [Sell] :" << endl;
    os << "prc : agg_sz : num_orders : feed_time : uuid" << endl;

    pub_side(ssagbk.sell_book_);

    return os;
}


auto l2_snappub1 = []<ExchID XID, FeedFormat FID, template<ExchID,FeedFormat>typename U>(const U<XID, FID>& tk, const SymbolState* pss)
                    {
                        if constexpr(is_same_v<decay_t<decltype(tk)>,QuoteLevelsOneSide<XID, FID>>)
                        {
                            cout << "snapshot : " << (static_cast<const SymbolStateAggPriceBook&>(*pss)) << endl;
                        }
                        else if constexpr(is_same_v<decay_t<decltype(tk)>,Symbol<XID, FID>>)
                        {
                            cout << "symbol : " << /* <<*/ endl;
                        }
                        else ///tmp
                        {
                            cout << type_name<remove_const_t<decltype(tk)>>() << endl;
                            cout << type_name<Symbol<XID, FID>>() << endl;
                            cout << "....................................KMW..................................." << endl;
                        }
                    };


SeqNumType pub_seq_num{};

template<ExchID XID, FeedFormat FID>
void FillOneSidedBookUpdateTick(AggBookIter& book_it, QuoteLevelsOneSide<XID,FID>& qlos)
{
    ///tmp
    qlos.seq_num = ++pub_seq_num;

    int i{};
    for(; i < qlos.num; ++i, ++book_it)
    {
        qlos.lvls[i] = book_it->second;
    }
}

auto l2_deltapub1 = []<ExchID XID, FeedFormat FID, template<ExchID,FeedFormat>typename U>(const U<XID, FID>& tk, const SymbolState*)
                     {
                         if constexpr(is_same_v<decay_t<decltype(tk)>,QuoteLevelsOneSide<XID, FID>>)
                         {
                            cout <<"qlos : " << tk << endl;
                         }
                         else /// tmp
                         {
                             cout << type_name<decltype(tk)>() << endl;
                             cout << type_name<Symbol<XID, FID>>() << endl;
                            cout << "....................................KMW..................................." << endl;
                         }
                     };


}

#endif // MKTDATASYSTEMPUB_H_INCLUDED

