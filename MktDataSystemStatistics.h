#ifndef MKTDATASYSTEMSTATISTICS_H_INCLUDED
#define MKTDATASYSTEMSTATISTICS_H_INCLUDED

#include "Definitions.h"
#include "MktDataSystemBaseDefinitions.h"
#include "MktDataSystemLogging.h"

namespace MKTDATASYSTEM::STATISTICS
{
using namespace MKTDATASYSTEM::DEFINITIONS;
using namespace MKTDATASYSTEM::LOGGING;

/// meant to be accumulated over an interval (eg, 1 minute)
/// ?? want to atomically exchange the whole LineStats object somehow - look into it
struct alignas(32) LineStats
{
    uint32_t pkts{};
    uint32_t msgs{};
    uint32_t bytes{};

    uint32_t pad[5]{};

    constexpr LineStats(){}

    void incr(const uint32_t ms, const uint32_t byts)
    {
        ++pkts;
        msgs += ms;
        bytes += byts;
    }
};

static_assert(sizeof(LineStats) == 32);

ostream& operator<<(ostream& os, const LineStats& ls)
{
    os << "[#pkts = " << ls.pkts << "] [#msgs = " << ls.msgs << "] [#bytes = " << ls.bytes << "]" << endl;
    return os;
}


/*
void linestats_exch_log(LineStats& ls)
{
    SRLOG(Level::INFO) << exch(ls);
}
*/
struct alignas(32) ProcessingStats
{
    uint32_t pkts;
    uint32_t msgs;
    uint64_t tm_spent;
    uint32_t pad[4];

    void incr(const uint32_t ms, const uint64_t tm_elapsed)
    {
        ++pkts;
        msgs += ms;
        tm_spent += tm_elapsed;
    }
};
static_assert(sizeof(ProcessingStats) == 32);

ostream& operator<<(ostream& os, const ProcessingStats& ps)
{
    os << "[#pkts = " << ps.pkts << "] [#msgs = " << ps.msgs << "] [#avg msg latency = " << ((ps.msgs ==0)?0.0:static_cast<double>(ps.tm_spent)/ps.msgs) << "]" << endl;
    return os;
}


/// not an atomic exchange !!
template<typename T>
requires (sizeof(T)==32 && alignof(T) == 32)
T exch(T& ls)
{
    T t;

    /// not an atomic exchange !!
    __m256i v{_mm256_load_si256(reinterpret_cast<__m256i*>(&ls))};/// hopefully an atomic load of all 32 bytes

    _mm256_store_si256(reinterpret_cast<__m256i*>(&ls), __m256i());

    memcpy(reinterpret_cast<void*>(&t), reinterpret_cast<void*>(&v), 32);

    return move(t);
}

}///namespace MKTDATASYSTEM::STATISTICS


namespace MKTDATASYSTEM::LOGGING
{
using namespace MKTDATASYSTEM::STATISTICS;

template<>
SRLog& SRLog::operator<< <LineStats>(const LineStats& ls)
{
    (*logStr_) << ls;
    return *this;
}

template<>
SRLog& SRLog::operator<< <ProcessingStats>(const ProcessingStats& ps)
{
    (*logStr_) << ps;
    return *this;
}

}//namespace MKTDATASYSTEM::LOGGING


namespace MKTDATASYSTEM::STATISTICS
{
using namespace MKTDATASYSTEM::DEFINITIONS;
using namespace MKTDATASYSTEM::LOGGING;

struct StatsCollector
{
    struct LineData
    {
        string name;
        LineStats ls;
    };
    deque<LineData> lndt;

    struct ProcessingData
    {
        string name;
        ProcessingStats ps;
    };
    deque<ProcessingData> procdt;
};

struct StatsUpdateObserver
{
    virtual void accept(StatsCollector&) = 0;
    virtual const char* obs_name() const = 0;
};

/// register a timing wake-up event
/// upon waking:
///     * apply StatsCollector to all registered observers
///     * observers will add to the collector in whatever category is appropriate
class StatsEventHandler
{
    StatsEventHandler(){}
    ~StatsEventHandler()
    {
        /// ??
    }

    /// block cp and assign and the move analogues
    StatsEventHandler(const StatsEventHandler&) = delete;
    StatsEventHandler& operator=(const StatsEventHandler&) = delete;

    inline static deque<StatsUpdateObserver*> obs_dq;

public:

    inline static thread evntThrd;

    static StatsEventHandler& inst()
    {
        static StatsEventHandler ob;
        return ob;
    }

    /// NOT a const* because the observer will zero out the existing stats data upon visiting it
    static void addsuo(StatsUpdateObserver* pso)
    {
        if(pso == nullptr) return;

        SRLOG(Level::STATS) << "adding StatsUpdateObserver : " << pso->obs_name();

        SRLOG(Level::STATS) << "kenny : ";
        obs_dq.push_back(pso);
    }


    void start(volatile bool& stop, const int interval_millis)
    {
        auto evntLoop = [&obs_dq, &stop]<typename clock = std::chrono::steady_clock>(const int interval_millis)
        {
            using duration = std::chrono::milliseconds;

            duration interval{interval_millis};

            std::chrono::time_point<clock> last_trigger_time{clock::now()};

            while(stop == false)
            {
                duration elapsed
                    = std::chrono::duration_cast<duration>(clock::now() - last_trigger_time);

                std::this_thread::sleep_for(interval - elapsed);

                cout << __FILE__ << " : "<< __LINE__ << " evntLoop 1" <<endl;
                StatsCollector sc;
                for(auto* p : obs_dq)
                {
                    p->accept(sc);
                }

                /// KMW: just look at the first one for now
                SRLOG(Level::STATS) << sc.lndt.front().name << " : " <<sc.lndt.front().ls;
                SRLOG(Level::STATS) << sc.procdt.front().name << " : " <<sc.procdt.front().ps;

                last_trigger_time = clock::now();
            }

        };

        evntThrd = move(thread(evntLoop,interval_millis));
    }
/*
    void onEvent()
    {
        StatsCollector sc;
        for(auto* p : obs_dq)
        {
            p->accept(sc);
        }
    }
*/
};

}

#endif // MKTDATASYSTEMSTATISTICS_H_INCLUDED
