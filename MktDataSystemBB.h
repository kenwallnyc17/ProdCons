#ifndef MKTDATASYSTEMBB_H_INCLUDED
#define MKTDATASYSTEMBB_H_INCLUDED

#include "MktDataSystemContainers.h"
#include "MktDataSystemTicks.h"
#include "MktDataSystemSymbolStates.h"
#include "MktDataSystemPub.h"
#include "MktDataSystemLogging.h"

namespace MKTDATASYSTEM::BOOKBUILDER
{
/// use MKTDATASYSTEM::TICK ticks (Add, Del, Can, Rep, Exec, ..) to maintain the aggregate price level book

using namespace MKTDATASYSTEM::DEFINITIONS;
using namespace MKTDATASYSTEM::TICK;
using namespace MKTDATASYSTEM::SYMBOLSTATES;

struct BookMod/// for publish
{
    AggBookIter it;/// start iter of book change
    uint8_t start_lvl;
    uint8_t num_lvls;/// number of levels changed from iter (including iter lvl)
};

template<Constants C>
struct BookBuilder
{
    optional<BookMod> operator()(const Add<C.XID,C.FID>&, Order* po);
    optional<BookMod> operator()(const Del<C.XID,C.FID>&, Order* po);
    optional<BookMod> operator()(const Can<C.XID,C.FID>&, Order* po);
    optional<BookMod> operator()(const Mod<C.XID,C.FID>&, Order* po);
    optional<BookMod> operator()(const Rep<C.XID,C.FID>&, Order* poo, Order* pno);
    optional<BookMod> operator()(const Exec<C.XID,C.FID>&, Order* po);
};

template<Constants C>
optional<BookMod> BookBuilder<C>::operator()(const Add<C.XID,C.FID>& tk, Order* po)
{
    /// kmw test
    if(po->sd == Side::Unknown) throw;

    ///?? does this lambda get reconstructed on each func call? -- without the static
    static auto update_book = [&tk](auto& book, Order* po) ->optional<BookMod>
    {
        SRLOG(Level::TRACE) << "BB Add order prc, sz = " << po->prc << ", " << po->sz;

        auto pr = book.try_emplace(po->prc, po->prc, po->sz, 1, 0, 0, nullptr);/// KMW --- deal with agg book iter

        if(pr.second == false)/// level already exists
        {
            auto& prclvl{pr.first->second};
            prclvl.agg_sz += po->sz;
            prclvl.num_orders += 1;
        }

        po->ab_iter = pr.first;

        pr.first->second.feed_time = tk.feed_time;
        pr.first->second.uuid = tk.uuid;

        int32_t lvl{distance(book.begin(), pr.first)};/// zero-based
        SRLOG(Level::TRACE) << "BB Add lvl = " << lvl;

        /// is at/near top of book
        if(lvl < MAX_PUB_LEVELS_TMP)[[likely]]
        {
            if(pr.second == false)/// level already exists
            {
                return BookMod{pr.first, static_cast<uint8_t>(lvl), 1};
            }
            else/// new prc lvl
            {
                auto max_book_lvls{(book.size()< MAX_PUB_LEVELS_TMP) ? uint8_t(book.size()) : MAX_PUB_LEVELS_TMP};
                return BookMod{pr.first, static_cast<uint8_t>(lvl), max_book_lvls - uint8_t(lvl)};
            }
        }

        return nullopt;
    };

    auto pss{static_cast<SymbolStateAggPriceBook*>(po->pss)};/// must be SymbolStateAggPriceBook*

    if(po->sd == Side::Buy)
    {
   // cout << __FILE__ << " : "<< __LINE__ << "buy book" <<endl;
        SRLOG(Level::TRACE) << "BB Add buy book";
        return update_book(pss->buy_book_, po);
    }
    else
    {
        //cout << __FILE__ << " : "<< __LINE__ << "sell book" <<endl;
        SRLOG(Level::TRACE) << "BB Add sell book";
        return update_book(pss->sell_book_, po);
    }

    return nullopt;/// KMW ??
}

template<Constants C>
optional<BookMod> BookBuilder<C>::operator()(const Del<C.XID,C.FID>& tk, Order* po)
{
    /// kmw test
    if(po->sd == Side::Unknown) throw;

    ///?? does this lambda get reconstructed on each func call? -- without the static
    static auto update_book = [&tk](auto& book, Order* po) ->optional<BookMod>
    {
        auto it{po->ab_iter};

        auto& prclvl{it->second};
        if(prclvl.num_orders > 1)
        {
            prclvl.agg_sz -= po->sz;
            prclvl.num_orders -= 1;
            prclvl.feed_time = tk.feed_time;
            prclvl.uuid = tk.uuid;
        }
        else
        {
            it = book.erase(it);
        }

        int32_t lvl{distance(book.begin(), it)};/// zero-based

        /// is at/near top of book
        if(lvl < MAX_PUB_LEVELS_TMP)[[likely]]
        {
            if(it == po->ab_iter)/// haven't deleted price level
            {
                return BookMod{it, static_cast<uint8_t>(lvl), 1};
            }
            else
            {
                auto max_book_lvls{(book.size()< MAX_PUB_LEVELS_TMP) ? uint8_t(book.size()) : MAX_PUB_LEVELS_TMP};
                return BookMod{it, static_cast<uint8_t>(lvl), max_book_lvls - lvl};
            }
        }

        return nullopt;
    };

    auto pss{static_cast<SymbolStateAggPriceBook*>(po->pss)};/// must be SymbolStateAggPriceBook*

    if(po->sd == Side::Buy)
    {
        return update_book(pss->buy_book_, po);
    }
    else
    {
        return update_book(pss->sell_book_, po);
    }

    return nullopt;/// KMW ??
}

template<Constants C>
optional<BookMod> BookBuilder<C>::operator()(const Can<C.XID,C.FID>& tk, Order* po)
{
    /// kmw test
    if(po->sd == Side::Unknown) throw;

    ///?? does this lambda get reconstructed on each func call? -- without the static
    static auto update_book = [&tk](auto& book, Order* po) ->optional<BookMod>
    {
        auto it{po->ab_iter};

        auto& prclvl{it->second};

    //don't do this    if(prclvl.agg_sz > tk.sz)/// ??? gaps can cause problems like this
        {
            SizeType size_to_rm{(po->sz > tk.sz) ? tk.sz : po->sz};

            if(po->sz > size_to_rm)
            {
                prclvl.agg_sz -= size_to_rm;
                po->sz -= size_to_rm; /// maybe better done in TickProcessor because this is the BB
            }
            else
            {
                if(prclvl.agg_sz > size_to_rm)
                {
                    prclvl.agg_sz -= size_to_rm;
                    prclvl.num_orders -= 1;
                }
                else
                {
                    prclvl.num_orders = 0; /// force level removal
                }

                po->sz = 0; /// remove order from Order container
            }
        }
   /*     else
        {
            prclvl.num_orders = 0; /// force level removal
        }
*/

        if(prclvl.num_orders == 0 )
        {
            it = book.erase(it);
        }
        else
        {
            prclvl.feed_time = tk.feed_time;
            prclvl.uuid = tk.uuid;
        }

        int32_t lvl{distance(book.begin(), it)};/// zero-based

        /// is at/near top of book
        if(lvl < MAX_PUB_LEVELS_TMP)[[likely]]
        {
            if(it == po->ab_iter)/// haven't deleted price level
            {
                return BookMod{it, static_cast<uint8_t>(lvl), 1};
            }
            else
            {
                auto max_book_lvls{(book.size()< MAX_PUB_LEVELS_TMP) ? uint8_t(book.size()) : MAX_PUB_LEVELS_TMP};
                return BookMod{it, static_cast<uint8_t>(lvl), max_book_lvls - lvl};
            }
        }

        return nullopt;
    };

    auto pss{static_cast<SymbolStateAggPriceBook*>(po->pss)};/// must be SymbolStateAggPriceBook*

    if(po->sd == Side::Buy)
    {
        return update_book(pss->buy_book_, po);
    }
    else
    {
        return update_book(pss->sell_book_, po);
    }

    return nullopt;/// KMW ??
}

/// what about side change?? NYSE might allow it (see spec)
template<Constants C>
optional<BookMod> BookBuilder<C>::operator()(const Mod<C.XID,C.FID>& tk, Order* po)
{
    /// kmw test
    if(po->sd == Side::Unknown) throw;

    ///?? does this lambda get reconstructed on each func call? -- without the static
    static auto update_book = [&tk](auto& book, Order* po) ->optional<BookMod>
    {
     //   to do:
     //  add ab_iter

        /// shouldn't occur
        if(po == nullptr )
        {
            assert(false);
        }

        auto oit{po->ab_iter};
        auto nit{oit};

        auto* poprclvl{&(oit->second)};///orig prc level
        auto* pnprclvl{poprclvl};/// new prc level - may update it below
        bool prc_changed{};

//    cout << __FILE__ << " : "<< __LINE__ << "  " << poprclvl->agg_sz << " : " << poo->sz <<endl;
//    cout << __FILE__ << " : "<< __LINE__ << "  " << poo << " : " << pno <<endl;
//    cout << __FILE__ << " : "<< __LINE__ << "  " << poo->prc << " : " << tk.new_prc <<endl;
        if(po->prc != tk.new_prc)
        {
            auto pr = book.try_emplace(tk.new_prc, tk.new_prc, tk.new_sz, 1, 0, 0, nullptr);

            pnprclvl = &pr.first->second;
        //cout << __FILE__ << " : "<< __LINE__ <<endl;
            if(pr.second == false)/// not a new prc level in the book
            {
        //cout << __FILE__ << " : "<< __LINE__ <<endl;
                pnprclvl->agg_sz += tk.new_sz;
                pnprclvl->num_orders += 1;
            }

            poprclvl->agg_sz -= po->sz;
            poprclvl->num_orders -= 1;
            nit = po->ab_iter = pr.first;
            po->prc = tk.new_prc;
            prc_changed = true;
        }
        else/// same prc, size change
        {
            pnprclvl->agg_sz -= po->sz;
            pnprclvl->agg_sz += tk.new_sz;
        }

        po->sz = tk.new_sz;

 //       }

#if 0
    //don't do this    if(prclvl.agg_sz > tk.sz)/// ??? gaps can cause problems like this
        {
            SizeType size_to_rm{(po->sz > tk.sz) ? tk.sz : po->sz};

            if(po->sz > size_to_rm)
            {
                prclvl.agg_sz -= size_to_rm;
                po->sz -= size_to_rm; /// maybe better done in TickProcessor because this is the BB
            }
            else
            {
                if(prclvl.agg_sz > size_to_rm)
                {
                    prclvl.agg_sz -= size_to_rm;
                    prclvl.num_orders -= 1;
                }
                else
                {
                    prclvl.num_orders = 0; /// force level removal
                }

                po->sz = 0; /// remove order from Order container
            }
        }
   /*     else
        {
            prclvl.num_orders = 0; /// force level removal
        }
*/
#endif

        /// so what if these are set twice sometimes!
        pnprclvl->feed_time = poprclvl->feed_time = tk.feed_time;
        pnprclvl->uuid = poprclvl->uuid = tk.uuid;

        if(poprclvl->num_orders == 0 )
        {
            oit = book.erase(oit);
        }

        int32_t lvl{(oit == nit) ? distance(book.begin(), oit) : min(distance(book.begin(), oit), distance(book.begin(), nit))};/// zero-based
        SRLOG(Level::TRACE) << "BB Mod : highest lvl modified (zero based) = " << lvl;

        /// is at/near top of book
        if(lvl < MAX_PUB_LEVELS_TMP)[[likely]]
        {
            auto it{book.begin()};
            advance(it, lvl);

            if(prc_changed == false)/// haven't deleted price level
            {
                return BookMod{it, static_cast<uint8_t>(lvl), 1};
            }
            else
            {
                /// ?? consider adding +1/*adding new lvl*/ to book.size()
                auto max_book_lvls{(book.size()< MAX_PUB_LEVELS_TMP) ? uint8_t(book.size()) : MAX_PUB_LEVELS_TMP};
                return BookMod{it, static_cast<uint8_t>(lvl), max_book_lvls - lvl};
            }
        }

        return nullopt;
    };

    auto pss{static_cast<SymbolStateAggPriceBook*>(po->pss)};/// must be SymbolStateAggPriceBook*

    if(po->sd == Side::Buy)
    {
        return update_book(pss->buy_book_, po);
    }
    else
    {
        return update_book(pss->sell_book_, po);
    }

    return nullopt;/// KMW ??
}


template<Constants C>
optional<BookMod> BookBuilder<C>::operator()(const Rep<C.XID,C.FID>& tk, /*SS* pss,*/ Order* poo, Order* pno)
{
    /// kmw test
    if(poo->sd == Side::Unknown) throw;

    ///?? does this lambda get reconstructed on each func call? -- without the static
    static auto update_book = [&tk](auto& book, Order* poo, Order* pno) ->optional<BookMod>
    {
        /// shouldn't occur
/*        handled in caller if(poo == pno )
        {
            assert(false);
        }
*/
        auto oit{poo->ab_iter};
        auto nit{oit};

        auto* poprclvl{&(oit->second)};///orig prc level
        auto* pnprclvl{poprclvl};/// new prc level - may update it below
        bool prc_changed{};

//    cout << __FILE__ << " : "<< __LINE__ << "  " << poprclvl->agg_sz << " : " << poo->sz <<endl;
//    cout << __FILE__ << " : "<< __LINE__ << "  " << poo << " : " << pno <<endl;
//    cout << __FILE__ << " : "<< __LINE__ << "  " << poo->prc << " : " << tk.new_prc <<endl;

        /// new order ID  OR   pno == poo (shouldn't happen)
 //       if(pno != nullptr)
//        {
        if(poo->prc != pno->prc)
        {
            auto pr = book.try_emplace(pno->prc, pno->prc, pno->sz, 1, 0, 0, nullptr);

            pnprclvl = &pr.first->second;

            if(pr.second == false)/// not a new prc level within the current book
            {
                pnprclvl->agg_sz += pno->sz;
                pnprclvl->num_orders += 1;
            }

            /// update old prc level
            poprclvl->agg_sz -= poo->sz;
            poprclvl->num_orders -= 1;
            nit = pno->ab_iter = pr.first;
            prc_changed = true;
        }
        else/// same prc, size change
        {
            /// ??? review
            pnprclvl->agg_sz -= poo->sz;
            pnprclvl->agg_sz += pno->sz;

            pno->ab_iter = oit;
        }

 //       }
#if 0
        else/// same order ID
        {
            if(poo->prc != tk.new_prc)
            {
                auto pr = book.try_emplace(tk.new_prc, tk.new_prc, tk.new_sz, 1, 0, 0, nullptr);

                pnprclvl = &pr.first->second;
//cout << __FILE__ << " : "<< __LINE__ <<endl;
                if(pr.second == false)/// not a new prc level
                {
 //cout << __FILE__ << " : "<< __LINE__ <<endl;
                    pnprclvl->agg_sz += tk.new_sz;
                    pnprclvl->num_orders += 1;
                }

                poprclvl->agg_sz -= poo->sz;
                poprclvl->num_orders -= 1;
                nit = poo->ab_iter = pr.first;
                poo->prc = tk.new_prc;
                poo->sz = tk.new_sz;
                prc_changed = true;
            }
            else/// same prc, size change
            {
                pnprclvl->agg_sz -= poo->sz;
                pnprclvl->agg_sz += tk.new_sz;
            }
        }
#endif
#if 0
    //don't do this    if(prclvl.agg_sz > tk.sz)/// ??? gaps can cause problems like this
        {
            SizeType size_to_rm{(po->sz > tk.sz) ? tk.sz : po->sz};

            if(po->sz > size_to_rm)
            {
                prclvl.agg_sz -= size_to_rm;
                po->sz -= size_to_rm; /// maybe better done in TickProcessor because this is the BB
            }
            else
            {
                if(prclvl.agg_sz > size_to_rm)
                {
                    prclvl.agg_sz -= size_to_rm;
                    prclvl.num_orders -= 1;
                }
                else
                {
                    prclvl.num_orders = 0; /// force level removal
                }

                po->sz = 0; /// remove order from Order container
            }
        }
   /*     else
        {
            prclvl.num_orders = 0; /// force level removal
        }
*/
#endif

        /// so what if it's set twice sometimes!
        pnprclvl->feed_time = poprclvl->feed_time = tk.feed_time;
        pnprclvl->uuid = poprclvl->uuid = tk.uuid;

        if(poprclvl->num_orders == 0 )
        {
            oit = book.erase(oit);
        }

        int32_t lvl{(oit == nit) ? distance(book.begin(), oit) : min(distance(book.begin(), oit), distance(book.begin(), nit))};/// zero-based
        SRLOG(Level::TRACE) << "BB Rep : highest lvl modified (zero based) = " << lvl;

        /// is at/near top of book
        if(lvl < MAX_PUB_LEVELS_TMP)[[likely]]
        {
            auto it{book.begin()};
            advance(it, lvl);

            if(prc_changed == false)/// haven't deleted price level
            {
                return BookMod{it, static_cast<uint8_t>(lvl), 1};
            }
            else
            {
                /// ?? consider adding +1/*adding new lvl*/ to book.size()
                auto max_book_lvls{(book.size()< MAX_PUB_LEVELS_TMP) ? uint8_t(book.size()) : MAX_PUB_LEVELS_TMP};
                return BookMod{it, static_cast<uint8_t>(lvl), max_book_lvls - lvl};
            }
        }

        return nullopt;
    };

    auto pss{static_cast<SymbolStateAggPriceBook*>(poo->pss)};/// must be SymbolStateAggPriceBook*

    if(poo->sd == Side::Buy)
    {
        return update_book(pss->buy_book_, poo, pno);
    }
    else
    {
        return update_book(pss->sell_book_, poo, pno);
    }

    return nullopt;/// KMW ??
}

}

namespace MKTDATASYSTEM::TICKPROCESSOR
{
//using namespace MKTDATASYSTEM;
using namespace MKTDATASYSTEM::DEFINITIONS;
using namespace MKTDATASYSTEM::TICK;
using namespace MKTDATASYSTEM::SYMBOLSTATES;
using namespace MKTDATASYSTEM::BOOKBUILDER;
using namespace MKTDATASYSTEM::PUB;

using FastOID = MKTDATASYSTEM::CONTAINERS::FastUIntHashMap<OrderID, Order*>;

/// default L3 tick processor just passes L3 ticks from Feed/Line Handlers to L3 publishers only

template<Constants C, derived_from<SymbolState> SS>
struct TickProcessor
{

protected:
    SymbolStateMgr<C,SS> ssMgr;
};

/**
template<ExchID XID, FeedFormat FID, typename... L3PUBS>
struct L3TickProcessor : TickProcessor
{
    /// NOTE: won't be able to specialize the below unless I also specialize the outer template --
    ///         https://en.cppreference.com/w/cpp/language/template_specialization.html
    template<template<ExchID, FeedFormat>typename T>
    requires MKTDATASYSTEM::TICK::IsL3Tick<T,XID,FID>
    void operator()(const T<XID,FID>& tk, L3PUBS&&... l3pubs)
    {
        (l3pubs(tk), ...);
    }

private:

};
*/
/// build agg order books from L3 input and publish L2 updates
template<Constants C, template<Constants>typename BB, typename...L2PUBS>
struct L2TickProcessor : TickProcessor<C,SymbolStateAggPriceBook>
{
    using symstate_t = SymbolStateAggPriceBook;

    void operator()(const Add<C.XID,C.FID>& tk);
    void operator()(const Del<C.XID,C.FID>& tk);
    void operator()(const Can<C.XID,C.FID>& tk);
    void operator()(Mod<C.XID,C.FID>& tk);
    void operator()(Rep<C.XID,C.FID>& tk);
    void operator()(const Symbol<C.XID,C.FID>& tk);

    L2TickProcessor() : foid(C.fastOIDMapInputs){}

private:
    BB<C> bb;
    FastOID foid;/// exch specific
    tuple<L2PUBS...> l2pubs_tpl;
};

/// new order
template<Constants C, template<Constants>typename BB, typename...L2PUBS>
void L2TickProcessor<C,BB,L2PUBS...>::operator()(const Add<C.XID,C.FID>& tk)
{
   /// try_emplace shouldn't fail on an Add order
    static auto f_onfail = [](const auto K, auto* V,auto&&...Vs)//const auto K, auto* V, const auto oid, const auto prc, const auto sz, const auto sd, auto bk_iter, auto*pss)
    {
        return;/// do nothing
    };

    symstate_t* pss{this->ssMgr.GetSymbolState(tk.symob)};
    if(pss == nullptr)
    {
        SRLOG(Level::ERR) << "L2TickProcessor Add: pss == nullptr ";
        ///log error
        return;
    }

    Order* pord{foid.try_emplace(tk.oid, /*f_onfail,*/ tk.oid, tk.prc, tk.sz, tk.side, AggBookIter(), pss)};/// default aggbookiter

    auto bkmod = bb(tk, pord);

    if constexpr(sizeof...(L2PUBS) > 0)
    {
        if (bkmod.has_value())
        {
          //  cout << __FILE__ << " : "<< __LINE__ << " start_lvl, num_lvls = " << int(bkmod.value().start_lvl) << ", " << int(bkmod.value().num_lvls) <<endl;
            MKTDATASYSTEM::TICK::QuoteLevelsOneSide<C.XID,C.FID> bk_delta(pss->symbol(), pss->prc_scale(), tk.side, bkmod.value().start_lvl,
                                                                      bkmod.value().num_lvls, tk.feed_time, tk.uuid);

            /// fill in bk_delta
            FillOneSidedBookUpdateTick(bkmod.value().it, bk_delta);

            std::apply([&bk_delta, pss](auto&&... f){(f(bk_delta, pss),...);}, l2pubs_tpl);
        }
    }
}

/// delete full existing order
template<Constants C, template<Constants>typename BB, typename...L2PUBS>
void L2TickProcessor<C,BB,L2PUBS...>::operator()(const Del<C.XID,C.FID>& tk)
{
    Order* pord{foid.find(tk.oid)};

    if(pord == nullptr)
    {
        SRLOG(Level::WARN) << "L2TickProcessor Del unknown orderid : " << tk.oid;
        return;
    }

    auto bkmod = bb(tk, pord);

    if constexpr(sizeof...(L2PUBS) > 0)
    {
        if (bkmod.has_value())
        {
            MKTDATASYSTEM::TICK::QuoteLevelsOneSide<C.XID,C.FID> bk_delta(pord->pss->symbol(), pord->pss->prc_scale(), pord->sd, bkmod.value().start_lvl,
                                                                      bkmod.value().num_lvls, tk.feed_time, tk.uuid);

            /// fill in bk_delta
            FillOneSidedBookUpdateTick(bkmod.value().it, bk_delta);

            std::apply([&bk_delta, pss = pord->pss](auto&&... f){(f(bk_delta, pss),...);}, l2pubs_tpl);
        }
    }

    foid.erase(tk.oid);
  //  delete pord;
}

/// delete portion of existing order
template<Constants C, template<Constants>typename BB, typename...L2PUBS>
void L2TickProcessor<C,BB,L2PUBS...>::operator()(const Can<C.XID,C.FID>& tk)
{
    Order* pord{foid.find(tk.oid)};

    if(pord == nullptr)
    {
        SRLOG(Level::WARN) << "L2TickProcessor Can unknown orderid : " << tk.oid;
        return;
    }

    if(tk.sz == 0)
    {
        SRLOG(Level::WARN) << "L2TickProcessor Can size 0 for orderid : " << tk.oid;
        return;
    }

    auto bkmod = bb(tk, pord);

    if constexpr(sizeof...(L2PUBS) > 0)
    {
        if (bkmod.has_value())
        {
            MKTDATASYSTEM::TICK::QuoteLevelsOneSide<C.XID,C.FID> bk_delta(pord->pss->symbol(), pord->pss->prc_scale(), pord->sd, bkmod.value().start_lvl,
                                                                      bkmod.value().num_lvls, tk.feed_time, tk.uuid);

            /// fill in bk_delta
            FillOneSidedBookUpdateTick(bkmod.value().it, bk_delta);

            std::apply([&bk_delta, pss = pord->pss](auto&&... f){(f(bk_delta, pss),...);}, l2pubs_tpl);
        }
    }

    if(pord->sz == 0)
    {
        delete pord;
    }
cout << __FILE__ << " : "<< __LINE__ << " reached end of Can " <<endl;

}

template<Constants C, template<Constants>typename BB, typename...L2PUBS>
void L2TickProcessor<C,BB,L2PUBS...>::operator()(Mod<C.XID,C.FID>& tk)
{
    Order* pord{foid.find(tk.orig_oid)};

    if(pord == nullptr)
    {
        SRLOG(Level::WARN) << "L2TickProcessor Mod unknown orderid : " << tk.orig_oid;
        return;
    }

    if(tk.new_sz == 0)
    {
        SRLOG(Level::WARN) << "L2TickProcessor Mod new size 0 for orderid : " << tk.orig_oid;
        return;
    }

    /// add valid prc and size to tick if defaults provided
    if(tk.new_prc == DefaultPrice)
        tk.new_prc = pord->prc;

    if(tk.new_sz == DefaultSize)
        tk.new_sz = pord->sz;

    auto bkmod = bb(tk, pord);

    if constexpr(sizeof...(L2PUBS) > 0)
    {
        if (bkmod.has_value())
        {
            MKTDATASYSTEM::TICK::QuoteLevelsOneSide<C.XID,C.FID> bk_delta(pord->pss->symbol(), pord->pss->prc_scale(), pord->sd, bkmod.value().start_lvl,
                                                                      bkmod.value().num_lvls, tk.feed_time, tk.uuid);

            /// fill in bk_delta
            FillOneSidedBookUpdateTick(bkmod.value().it, bk_delta);

            std::apply([&bk_delta, pss = pord->pss](auto&&... f){(f(bk_delta, pss),...);}, l2pubs_tpl);
        }
    }

/*    if(pord->sz == 0)
    {
        delete pord;
    }
*/
cout << __FILE__ << " : "<< __LINE__ << " reached end of Mod " <<endl;

}

template<Constants C, template<Constants>typename BB, typename...L2PUBS>
void L2TickProcessor<C,BB,L2PUBS...>::operator()(Rep<C.XID,C.FID>& tk)
{
    Order* poord{foid.find(tk.orig_oid)};

    /// need orig order: need side and maybe size and price
    if(poord == nullptr)
    {
        SRLOG(Level::WARN) << "L2TickProcessor Rep unknown orig orderid : " << tk.orig_oid;
        return;
    }

/// KMW note: Replace orders should always provide new order id.. no longer combining Mod and Replace into a Replace tick
    if(tk.new_oid == DefaultOrderID || tk.new_oid == 0 || tk.orig_oid == tk.new_oid)
    {
        SRLOG(Level::ERR) << "L2TickProcessor Rep invalid new orderid : " << tk.new_oid;
        return;
    }

//cout << __FILE__ << " : "<< __LINE__ << "  " << poord->prc << " : " << tk.new_prc <<endl;
//cout << __FILE__ << " : "<< __LINE__ << "  " << poord->sz << " : " << tk.new_sz <<endl;

    /// add valid prc and size to tick if defaults provided
    if(tk.new_prc == DefaultPrice)
        tk.new_prc = poord->prc;

    if(tk.new_sz == DefaultSize)
        tk.new_sz = poord->sz;

//cout << __FILE__ << " : "<< __LINE__ << "  " << poord->prc << " : " << tk.new_prc <<endl;
//cout << __FILE__ << " : "<< __LINE__ << "  " << poord->sz << " : " << tk.new_sz <<endl;


    Order* pnord{foid.try_emplace(tk.new_oid, tk.new_oid, tk.new_prc, tk.new_sz,
                                 poord->sd, AggBookIter(), poord->pss)};

    /// probably never happens -- only order id change
    if(tk.new_prc == poord->prc && tk.new_sz == poord->sz)
    {
        if(tk.orig_oid != tk.new_oid)
        {
            delete poord;/// have created new ord which will be used to refer to this prc and sz combo which is unchanged
        }

        assert(false);/// test

        return;
    }

    auto bkmod = bb(tk, poord, pnord);

    if constexpr(sizeof...(L2PUBS) > 0)
    {
        if (bkmod.has_value())
        {
            MKTDATASYSTEM::TICK::QuoteLevelsOneSide<C.XID,C.FID> bk_delta(poord->pss->symbol(), poord->pss->prc_scale(), poord->sd, bkmod.value().start_lvl,
                                                                      bkmod.value().num_lvls, tk.feed_time, tk.uuid);

            /// fill in bk_delta
            FillOneSidedBookUpdateTick(bkmod.value().it, bk_delta);

            std::apply([&bk_delta, pss = poord->pss](auto&&... f){(f(bk_delta, pss),...);}, l2pubs_tpl);
        }
    }

    if(tk.orig_oid != tk.new_oid)
    {
        foid.erase(tk.orig_oid);
    }

/*    if(pord->sz == 0)
    {
        delete pord;
    }
*/
    SRLOG(Level::TRACE) << "L2TickProcessor : reached end of Rep ";
}

template<Constants C, template<Constants>typename BB, typename...L2PUBS>
void L2TickProcessor<C,BB,L2PUBS...>::operator()(const Symbol<C.XID,C.FID>& tk)
{
    /// add symbol to ssMgr
    symstate_t* pss{this->ssMgr.GetSymbolState({tk.sym, tk.sym_idx, tk.prc_scale})};

    if(pss == nullptr)
    {
        SRLOG(Level::ERR) << "L2TickProcessor Symbol: pss == nullptr ";
        ///log error
        return;
    }

    if constexpr(sizeof...(L2PUBS) > 0)
    {
        std::apply([&tk, pss /*= nullptr*/](auto&&... f){(f(tk, pss),...);}, l2pubs_tpl);
    }

    SRLOG(Level::TRACE) << "L2TickProcessor : reached end of Symbol ";
}

}

#endif // MKTDATASYSTEMBB_H_INCLUDED
