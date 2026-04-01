#ifndef MKTDATASYSTEMINPUTS_H_INCLUDED
#define MKTDATASYSTEMINPUTS_H_INCLUDED

#include "MktDataSystemBaseDefinitions.h"
#include "MktDataSystemStatistics.h"
#include "MktDataSystemContainers.h"
#include "MktDataSystemTesting.h"

namespace MKTDATASYSTEM::INPUTS
{
using namespace MKTDATASYSTEM::DEFINITIONS;
using namespace MKTDATASYSTEM::STATISTICS;
using namespace MKTDATASYSTEM::CONTAINERS;

/**
PhysLineI(A/B/C/...)
PhysLineI(A/B/C/...)
...
PhysLineI(A/B/C/...)
                        --> LineI (logical)
*/

struct PhysLine : LineHdr/// A/B/C/...
{
 //   char name[max_line_name_sz+1];/// maybe channel # + A/B/C

  //  constexpr PhysLine(const uint32_t ip, const uint16_t port, const char* nm) : LineHdr(ip,port), stats() {set_line_name(nm, name);}

    void visit(const string& line, const int i, StatsCollector& sc) const
    {
        string physline(line);
        physline += {'_','A'+i};
        sc.lndt.emplace_back(physline, exch(stats));
    }

    mutable LineStats stats;

 //  Packet ab_pkt;/// tmp
};

using PhysLinePtr = PhysLine*;

/// handle more than 2 physical lines for logical line
/// prefer to use array for phys_lines_dq, but, then size is compile time req
struct Line /// logical line -- arbitrated line
{
  //  constexpr static size_t max_line_name_chars{8};
    constexpr static size_t max_phys_lines{3};
    Line() = default;
    constexpr Line(const char* nm/*string& name*/, /*initializer_list<PhysLinePtr> il*/auto... ptrs) requires (sizeof...(ptrs) < max_phys_lines)
        : phys_lines_arr({ptrs...})
    {
        set_line_name(nm, name);
      /*  for(auto& el : il)
        {
            phys_lines_dq[] = el;
        }
    */
    }

    constexpr Line(const Line&) = default;

//--    /// join the multicast groups defined in phys_lines_dq
//--    bool joinGroups();

    void visit(StatsCollector& sc) const
    {
        sc.lndt.emplace_back(name, exch(stats));

        int i{};
        for(auto el : phys_lines_arr)
        {
            if(el == nullptr)
                continue;

            el->visit(name, i++, sc);
        }
    }

    void set_arb_pkt(Packet* pkt) const
    {
        p_arb_pkt = pkt;
    }

    /// name
    char name[max_line_name_sz+1];
    mutable Packet* p_arb_pkt;/// arbitrated pkt
  //  PhysLinePtr pArb_line{};

    /*deque<PhysLinePtr>*/ array<const PhysLine*,max_phys_lines> phys_lines_arr;

    mutable SeqNumType next_expected_seq_num{};
    mutable LineStats stats;
};

using LinePtr = Line*;

template<typename PL>
concept IsLinePtr = (is_pointer_v<PL> && derived_from<remove_pointer<PL>, Line>);

struct PhysLineAttributes
{
    const Line* p_logical_line{};
    const PhysLine* p_phys_line{};
//    Packet* p_pkt_buffer;///???
};

template<size_t NPhysLines>
struct LineArbitration
{
    void add(const Line* pl)
    {
        for(auto* ppl : pl->phys_lines_arr)
        {
          //  cout << __FILE__ << " : "<< __LINE__ << " LineArbitration::add " << ppl <<endl;

            if(ppl == nullptr)
                continue;

            phys_line_attr_m.insert({ppl->mc_ip, ppl->mc_port},{pl, ppl /*, nullptr*/});
            //phys_line_attr_m.insert({{ppl->mc_ip, ppl->mc_port},{pl, ppl, nullptr}});
        }
    }

    const PhysLineAttributes* getAttributes(const Packet& pkt)
    {
        return phys_line_attr_m.find({pkt.line.mc_ip, pkt.line.mc_port});
    }
    SockAddrFlatMap<uint32_t, PhysLineAttributes, NPhysLines> phys_line_attr_m;
};


struct RecvInterface
{
    virtual bool init() = 0;
    virtual bool join(const PhysLinePtr) = 0;/// call for each group
    virtual bool read(const Line*& lp) = 0;

    RecvInterface(const deque<const Line*>& lines_dq, const size_t proc) : lines_dq(lines_dq), proc_(proc) {}
    const deque<const Line*>& lines_dq;

protected:
    const size_t proc_;
};


/// join -> register for recv of certain physical MC groups
/// read :
///     * read from single ring buffer which contains all line physical MC group packets
///     * update Phys line stats
///     * per logical line, arbitrate btw associated phys lines
///         -> if gap, buffer packets
///         -> else, pass first correctly sequenced packet through to processing
///     * use deque's for now, later use more efficient queues
template<typename RAWPACKET, size_t NTickProcs, size_t NPhysLines>
struct GenericKernelBypassRecvInterface : RecvInterface
{
    GenericKernelBypassRecvInterface(const deque<const Line*>& lines_dq, const size_t proc)
        : RecvInterface(lines_dq, proc) {}

    bool init()
    {
         /// init efvi lib, etc.

        for(auto& el : lines_dq)
            line_arb.add(el);

      //  inpkts_alllines_q = EXAMPLEINPUTS::MultiLineStingRayTest<1, NTickProcs>::inst().getProcPktStream(proc_);

        auto& pktstrm{EXAMPLEINPUTS::MultiLineStingRayTest<1, NTickProcs>::inst().getProcPktStream(proc_)};

        cout << __FILE__ << " : "<< __LINE__ << " pktstrm info : " << proc_ << ", " << pktstrm.size() << endl;
        for(auto& el : pktstrm)
            inpkts_alllines_q.push_back(el);

        cout << __FILE__ << " : "<< __LINE__ << " after build and copy into inpkts_alllines_q" <<endl;

        return true;
    }

    bool join(const PhysLinePtr ppl)
    {
        /// efvi.join(ppl->IP, ppl->port)
        return true;
    }

    bool read(const Line*& lp)
    {
        enum class PktAction{Analyzing, Process, Skip, Store};

        PktAction pktAct{PktAction::Analyzing};
        do
        {
            /// likely scenario
            ///     * the packet at the front is the last packet sent for processing and has already been analyzed
            if(inpkts_alllines_q.empty() == false
             //  && inpkts_alllines_q.front().has_been_analyzed == true)
               && last_packet.has_been_analyzed == true)
            {
                inpkts_alllines_q.pop_front();
            }

            cout << __FILE__ << " : "<< __LINE__ << " in read before while" <<endl;
            int cnt{};
            while(inpkts_alllines_q.empty() == true)
            {
                ++cnt;
            }

            cout << __FILE__ << " : "<< __LINE__ << " in read after while : "
                << inpkts_alllines_q.size() << ", " << cnt <<endl;

          //  Packet pkt{inpkts_alllines_q.front()};
            last_packet = move(convertPkt(inpkts_alllines_q.front()));

            cout << __FILE__ << " : "<< __LINE__ << " in read after convertPkt : " << endl;

/// this shouldn't be necessary... should only be max one packet at front that has been analyzed
 /*           if(pkt.has_been_analyzed == true)
            {
                inpkts_alllines_q.pop_front();
                continue;
            }
*/
            const PhysLineAttributes* pAttr {line_arb.getAttributes(last_packet)};

            const Line* pLogicalLine{pAttr->p_logical_line};
            const PhysLine* pPhysicalLine{pAttr->p_phys_line};

        cout << __FILE__ << " : "<< __LINE__ << " in read after getAttributes : "
            << pAttr << ", " << pLogicalLine << ", " << pPhysicalLine << endl;


            pPhysicalLine->stats.incr(last_packet.msg_cnt, last_packet.len);

            if(pLogicalLine->next_expected_seq_num == last_packet.first_seq_num || pLogicalLine->next_expected_seq_num == 0)
            {
                pLogicalLine->stats.incr(last_packet.msg_cnt, last_packet.len);
                pLogicalLine->next_expected_seq_num += last_packet.msg_cnt;
//here               lp->pArb_line->ab_pkt = move(pkt);/// ?? make this a ptr to the pkt sitting in the queue

               // lp->p_arb_pkt = &pkt;
                lp->set_arb_pkt(&last_packet);

                pktAct = PktAction::Process;
            }
            else if(pLogicalLine->next_expected_seq_num > last_packet.first_seq_num)
            {
               ///skip packet
               pktAct = PktAction::Skip;
               inpkts_alllines_q.pop_front();
            }
            else[[unlikely]]
            {
               /// store packet and wait for the missing packet for some time before declaring a gap
               pktAct = PktAction::Store;

//here               /// store it

               inpkts_alllines_q.pop_front();
            }

            last_packet.has_been_analyzed = true;

          //  inpkts_alllines_q.pop_front();

        }while(pktAct != PktAction::Process);
    }
//FakeRawFullPacket<EXCHHEADER>
    deque<RAWPACKET> inpkts_alllines_q;
   // deque<Packet> inpkts_alllines_q;/// theoretically, kernel bypass system pushes packets into this

    Packet last_packet;
    LineArbitration<NPhysLines> line_arb;
};

struct EFVIRecvInterface : RecvInterface
{
    bool init()
    {
         /// init efvi lib, etc.
         return true;
    }

    bool join(const PhysLinePtr ppl)
    {
        /// efvi.join(ppl->IP, ppl->port)
        return true;
    }

    bool read(LinePtr& lp)
    {
       // return efvi.read_next(pkt.data);
    }
};

struct FileReadInterface : RecvInterface
{
    bool init()
    {
         ///
         return true;
    }

    bool join(const PhysLinePtr ppl)
    {
        ///
        return true;
    }

    bool read(LinePtr& lp)
    {
        return true;
    }
};

struct QueueReadInterface : RecvInterface
{
    bool init()
    {
         ///
         return true;
    }

    bool join(const PhysLinePtr ppl)
    {
        ///
        return true;
    }

    bool read(LinePtr& lp)
    {
        return true;
    }

};

}

#endif // MKTDATASYSTEMINPUTS_H_INCLUDED
