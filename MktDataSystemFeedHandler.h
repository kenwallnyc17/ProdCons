#ifndef MKTDATASYSTEMFEEDHANDLER_H_INCLUDED
#define MKTDATASYSTEMFEEDHANDLER_H_INCLUDED

#include "MktDataSystemInputs.h"
#include "MktDataSystemBB.h"
#include "MktDataSystemLogging.h"
#include "MktDataSystemStatistics.h"


namespace MKTDATASYSTEM::FEEDHANDLER
{
using namespace MKTDATASYSTEM::DEFINITIONS;
using namespace MKTDATASYSTEM::INPUTS;
using namespace MKTDATASYSTEM::TICKPROCESSOR;


/// base class for decoding a packet
/// a concrete instance will process a group of lines and have it's own core assigned to it

struct LineHandler : StatsUpdateObserver
{
    LineHandler(const string& name, initializer_list<const Line*> il) : name(name), lines_dq(il) {}

    /// initialization:
    ///     * create interface, join groups
    ///     * ?? taskset this handler to a specific core
/*    bool init(RecvInterface& ifc)
    {
        ifc.init();

        for(auto& el : lines_dq)
        {
            ///ifc.join(el);
        }

        return true;
    }
*/
    virtual bool process(const Line* lp) = 0;

    void accept(StatsCollector& sc) override
    {
        for(auto el : lines_dq)
        {
            el->visit(sc);
        }

        sc.procdt.emplace_back(name, exch(proc_stats));
    }

    const char* obs_name() const override{return name.c_str();}

    const string name;
    deque<const Line*> lines_dq;
    ProcessingStats proc_stats;
};
/*
template<derived_from<TickProcessor> TP>
struct Itch5LineHandler : LineHandler
{
    Itch5LineHandler(initializer_list<LinePtr> il) : LineHandler(il) {}

    /// loop over packet processing it's contained msgs -- can do a
    bool process(Packet&) override{}

    bool add(const PktMsg& msg);
    bool rep(const PktMsg& msg);
    bool can(const PktMsg& msg);
    bool exec(const PktMsg& msg);
    bool status(const PktMsg& msg);

private:
    TP tp;
};
*/
template<Constants C,
        derived_from<RecvInterface> RCVTYPE,
        template<Constants> typename BB,
        template<Constants,template<Constants>class, class...> typename TPTYPE,
        template<Constants,class> typename LHTYPE, typename...L2PUBS>
        requires (derived_from<TPTYPE<C,BB,L2PUBS...>, TickProcessor<C,typename TPTYPE<C,BB,L2PUBS...>::symstate_t>>
                  && derived_from<LHTYPE<C,TPTYPE<C,BB,L2PUBS...>>,LineHandler>)
struct FeedHandler
{
    std::thread start_proc(auto&& runF, const int core, const size_t proc, auto... lnptrs)
    {
        cout << __FILE__ << " : "<< __LINE__ << " start_proc " <<endl;
        LnHndlr_t* pLH = createLH({lnptrs...});

        Rcvr_t* pRcvr = createRcvr(pLH, proc);

        pRcvr->init();/// join groups, load file, ...`

        cout << __FILE__ << " : "<< __LINE__ << " start_proc call runF" <<endl;
        return move(std::thread(runF, core, pLH, pRcvr));
    }
private:
    using LnHndlr_t = LHTYPE<C,TPTYPE<C,BB,L2PUBS...>>;
    using Rcvr_t = RCVTYPE;

    deque<LnHndlr_t*> deq_pLH{};
    deque<Rcvr_t*> deq_pRcvr{};

    LnHndlr_t* createLH(initializer_list<const Line*> il)//*/auto&&... lptrs)
    {
        static int cnt{};
        LnHndlr_t* pLH = new LnHndlr_t("LineHndlr_" + to_string(++cnt),il);//lptrs...);
        deq_pLH.push_back(pLH);
        return pLH;
    }

    Rcvr_t* createRcvr(const LnHndlr_t* pLH, const size_t proc)
    {
        Rcvr_t* pRcvr = new Rcvr_t(pLH->lines_dq, proc);
        deq_pRcvr.push_back(pRcvr);
        return pRcvr;
    }
};

/*
struct Itch5FeedHandler : FeedHandler<FeedFormat::ITCH5, Itch5LineHandler>
{
    ///

};
*/
/*
volatile bool run{};


/// taskset an invocation of this to a specific core
auto LineProcessor = []<derived_from<RecvInterface> IFACE, derived_from<LineHandler> LHTYPE>(IFACE& ifc, LHTYPE& lhndlr)
{
    lhndlr.init(ifc);

    while(run)
    {
        Packet pkt;
        if(ifc.read(pkt) == true)
        {
            lhndlr.process(pkt);
        }
    }
};
*/



}

#endif // MKTDATASYSTEMFEEDHANDLER_H_INCLUDED
