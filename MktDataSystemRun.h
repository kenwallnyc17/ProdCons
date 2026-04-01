#ifndef MKTDATASYSTEMRUN_H_INCLUDED
#define MKTDATASYSTEMRUN_H_INCLUDED

#include "MktDataSystemFeedHandler.h"
#include "MktDataSystemStingRayFeedHandler.h"
#include "MktDataSystemNYSEIntegratedFeedHandler.h"
#include "StingRayPub.h"
#include "MktDataSystemTesting.h"
#include "MktDataSystemLogging.h"
#include "PerfAnalysis.h"

//inline constexpr bool BuildStingRay{true};
inline constexpr bool BuildStingRay{false};

inline constexpr bool BuildMultiLineStingRay{true};

//inline constexpr bool BuildARCA{true};
inline constexpr bool BuildARCA{false};

bool startFeedHandler()//int bkts)//const Config& cfg)
{
    Logger::inst().init("D:\\log.out", Level::TRACE, 32*1024);
    Logger::inst().start();
    //setLogThreshold()

   // using namespace MKTDATASYSTEM::LOGGING;
#if 0
    if(false)
    {
      //  SRLog logr(__FILE__, __LINE__, MKTDATASYSTEM::LOGGING::Level::INFO);

        SRLOG(Level::INFO) << "test kmddddddddddddddddddddddddddddddddddddddddddddddddddd\n";
        SRLOG(Level::INFO) << "test kwalllllllllllllllllllllllllllllllllllllllll\n";
    }
    {
        int i {18};

        SRLOG(Level::WARN) << "test warn : " << i;
    }

    {
        int i {19};

        SRLOG(Level::WARN) << "test warn : " << i << "\n";
    }

    {
        int i {20};

        SRLOG(Level::INFO) << "test info : " << i << "kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk\n";
    }
#endif // 0

  /*  Logger::inst().stop();

    int k;
    cin >> k;
    return true;
*/
    /// read from config:
    /// exch, format, input type, book build type,
    /// tick processing type with matching pub type : L3 ticks can have L2/L3 pub
    /// line handler type should match tick type

    /// read lines: multicast/files/etc.

    /// for test use these

    /// this is here because a conflict with the STINGRAY namespace being both under
    ///   MKTDATASYSTEM::FEEDHANDLER and a main namespace at same level as MKTDATASYSTEM
    using SRRawPacketType = typename EXAMPLEINPUTS::FakeRawFullPacket<STINGRAY::SRPacketHeader>;

    using namespace MKTDATASYSTEM::DEFINITIONS;
    using namespace MKTDATASYSTEM::SYMBOLSTATES;
    using namespace MKTDATASYSTEM::FEEDHANDLER;
    using namespace MKTDATASYSTEM::TICKPROCESSOR;
    using namespace MKTDATASYSTEM::BOOKBUILDER;
    using namespace MKTDATASYSTEM::PUB;
    using namespace MKTDATASYSTEM::INPUTS;
    using namespace MKTDATASYSTEM::STATISTICS;

    volatile bool stop{false};

    StatsEventHandler& statsEvents{StatsEventHandler::inst()};

    deque<thread> proc_threads;

    static auto proc_loop = [&stop, &statsEvents](int core, LineHandler* pLH, RecvInterface* pRcvr)
    {
        cout << __FILE__ << " : "<< __LINE__ << " proc_loop before add to statsevent" <<endl;
        ::SetThreadAffinityMask(::GetCurrentThread(), 1 << core);

    //    pRcvr->init();/// join groups, load file, ...
        statsEvents.addsuo(pLH);

        stopwatch_hr tmr;

        const Line* lp{};

        while(stop == false)
        {
            if(pRcvr->read(lp))
            {
                tmr.start();
              //  std::this_thread::sleep_for(200ms);
                pLH->process(lp);
                tmr.stop();

                lp->stats.incr(lp->p_arb_pkt->msg_cnt, lp->p_arb_pkt->len);

                pLH->proc_stats.incr(lp->p_arb_pkt->msg_cnt, tmr.latency());

                SRLOG(Level::INFO) << "avg_lat : " << tmr.avg_latency();

                tmr.reset();

            }

           // SRLOG(Level::INFO) << "stopwatch stats : " << tmr.avg_latency();
        }

        return true;
    };

/// above is generic code
///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#if 1
    if constexpr (BuildStingRay == true)
    {
        constexpr auto xid = ExchID::STINGRAY;
        constexpr auto fid = FeedFormat::STINGRAYBINARY;

        constexpr Constants ConstsForBuild{xid,fid};

        constexpr PhysLine pln_1A({12345, 5001});
        const PhysLine pln_1B({12345, 5002});
        const PhysLine pln_2A({12346, 5001});
        const PhysLine pln_2B({12346, 5002});

        const Line ln_1("line1", &pln_1A, &pln_1B);
        const Line ln_2("line2", &pln_2A, &pln_2B);

        /// custom runtime setting from config
      //  ConstsForBuild.fastOIDMapInputs = FastHashMapInputs{bkts,4};

        using input = typename EXAMPLEINPUTS::StingRayReadInterface;

        FeedHandler<ConstsForBuild,input,MKTDATASYSTEM::BOOKBUILDER::BookBuilder,
                        L2TickProcessor,StingRayLineHandler,decltype(l2_deltapub1),decltype(l2_snappub1)> fh;


        int core{22};
        thread t1 = fh.start_proc(proc_loop, core, 0, &ln_1, &ln_2);

        proc_threads.emplace_back(move(t1));
    }

    if constexpr(BuildMultiLineStingRay == true)
    {
        constexpr auto xid = ExchID::STINGRAY;
        constexpr auto fid = FeedFormat::STINGRAYBINARY;

        constexpr Constants ConstsForBuild{xid,fid};

        constexpr size_t num_tick_procs{2};
        constexpr size_t num_lines_tp1{2};
        constexpr size_t num_lines_tp2{2};
        constexpr size_t num_lines{num_lines_tp1+num_lines_tp2};
        constexpr size_t num_phys_lines_per{2};
        constexpr size_t max_phys_lines_per_rcvr{num_lines_tp1*num_phys_lines_per};

        using input = GenericKernelBypassRecvInterface<SRRawPacketType,
                        num_tick_procs, max_phys_lines_per_rcvr>;

        FeedHandler<ConstsForBuild,input,MKTDATASYSTEM::BOOKBUILDER::BookBuilder,
                        L2TickProcessor,StingRayLineHandler,decltype(l2_deltapub1),decltype(l2_snappub1)> fh;


        constexpr PhysLine phys_lines_arr[num_lines][num_phys_lines_per] =
        {
            {PhysLine({inet_addr_constexpr("224.0.59.0"), 23200}), PhysLine({inet_addr_constexpr("224.0.59.128"), 23700})},
            {PhysLine({inet_addr_constexpr("224.0.59.1"), 23201}), PhysLine({inet_addr_constexpr("224.0.59.129"), 23701})},
            {PhysLine({inet_addr_constexpr("224.0.59.2"), 23202}), PhysLine({inet_addr_constexpr("224.0.59.130"), 23702})},
            {PhysLine({inet_addr_constexpr("224.0.59.3"), 23203}), PhysLine({inet_addr_constexpr("224.0.59.131"), 23703})}
        };

        const Line lines_arr_tp1[num_lines_tp1] =
        {
            Line("Line1", &phys_lines_arr[0][0], &phys_lines_arr[0][1]),
            Line("Line2", &phys_lines_arr[1][0], &phys_lines_arr[1][1])
        };

        int core{22};
        thread t1 = fh.start_proc(proc_loop, core, 0, &lines_arr_tp1[0], &lines_arr_tp1[1]);

        proc_threads.emplace_back(move(t1));

        const Line lines_arr_tp2[num_lines_tp2] =
        {
            Line("Line3", &phys_lines_arr[2][0], &phys_lines_arr[2][1]),
            Line("Line4", &phys_lines_arr[3][0], &phys_lines_arr[3][1])
        };

 //       int v;
 //   cin >> v;

        core = 21;
        thread t2 = fh.start_proc(proc_loop, core, 1, &lines_arr_tp2[0], &lines_arr_tp2[1]);
        proc_threads.emplace_back(move(t2));

        statsEvents.start(stop, 200);

        for(auto& thrd : proc_threads)
            thrd.join();

    cout << __FILE__ << " : "<< __LINE__ << " after joins" <<endl;
    SRLOG(Level::INFO) << " after joins";

      //  std::this_thread::sleep_for(30000ms);
    }

    if constexpr(BuildARCA == true)
    {
        constexpr auto xid = ExchID::NYSEARCA;
        constexpr auto fid = FeedFormat::NYSEINTEGRATED;

        constexpr Constants ConstsForBuild{xid,fid};

        constexpr size_t num_tick_procs{2};
        constexpr size_t num_lines_tp1{2};
        constexpr size_t num_lines_tp2{2};
        constexpr size_t num_lines{num_lines_tp1+num_lines_tp2};
        constexpr size_t num_phys_lines_per{2};
        constexpr size_t max_phys_lines_per_rcvr{num_lines_tp1*num_phys_lines_per};

        using input = GenericKernelBypassRecvInterface<SRRawPacketType,
                        num_tick_procs,max_phys_lines_per_rcvr>;

        FeedHandler<ConstsForBuild,input,MKTDATASYSTEM::BOOKBUILDER::BookBuilder,
                        L2TickProcessor,NYSEIntegratedLineHandler,decltype(l2_deltapub1),decltype(l2_snappub1)> fh;


        constexpr PhysLine phys_lines_arr[num_lines][num_phys_lines_per] =
        {
            {PhysLine({inet_addr_constexpr("224.0.59.0"), 23200}), PhysLine({inet_addr_constexpr("224.0.59.128"), 23700})},
            {PhysLine({inet_addr_constexpr("224.0.59.1"), 23201}), PhysLine({inet_addr_constexpr("224.0.59.129"), 23701})},
            {PhysLine({inet_addr_constexpr("224.0.59.2"), 23202}), PhysLine({inet_addr_constexpr("224.0.59.130"), 23702})},
            {PhysLine({inet_addr_constexpr("224.0.59.3"), 23203}), PhysLine({inet_addr_constexpr("224.0.59.131"), 23703})}
        };

        const Line lines_arr_tp1[num_lines_tp1] =
        {
            Line("Line1", &phys_lines_arr[0][0], &phys_lines_arr[0][1]),
            Line("Line2", &phys_lines_arr[1][0], &phys_lines_arr[1][1])
        };

        int core{22};
        thread t1 = fh.start_proc(proc_loop, core, 0, &lines_arr_tp1[0], &lines_arr_tp1[1]);

        proc_threads.emplace_back(move(t1));

        const Line lines_arr_tp2[num_lines_tp2] =
        {
            Line("Line3", &phys_lines_arr[2][0], &phys_lines_arr[2][1]),
            Line("Line4", &phys_lines_arr[3][0], &phys_lines_arr[3][1])
        };

 //       int v;
 //   cin >> v;

        core = 21;
        thread t2 = fh.start_proc(proc_loop, core, 1, &lines_arr_tp2[0], &lines_arr_tp2[1]);
        proc_threads.emplace_back(move(t2));
    }
#endif


    /// start the Stats handler
   // statsEvents.start(stop, 200);

    int v;
    cin >> v;

    stop = true;


    Logger::inst().stop();

    cin >> v;

    statsEvents.evntThrd.join();



    return true;
}

#endif // MKTDATASYSTEMRUN_H_INCLUDED
