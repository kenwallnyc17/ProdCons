#ifndef STINGRAYPUB_H_INCLUDED
#define STINGRAYPUB_H_INCLUDED

#include "MktDataSystemInputs.h"
#include "StingRayExch.h"

namespace EXAMPLEINPUTS
{
using namespace STINGRAY;

template<uint8_t ver, typename...Msgs>
uint16_t buildPacket(char* buf, uint32_t& msg_seq_num, Msgs&&... msgs)
{
    uint16_t len{sizeof(SRPacketHeader)};

    static auto pack = [&buf, &len](auto&& msg)
    {
        memcpy(buf+len, reinterpret_cast<void*>(&msg), sizeof(msg));
        len += sizeof(msg);
    };

    (..., pack(msgs));

    SRPacketHeader hdr{msg_seq_num, len, ver, sizeof...(msgs)};
//    hdr.msg_cnt = 2;//sizeof...(msgs);
//cout << __FILE__ << " : "<< __LINE__ << " : " << hdr.msg_seq_num<< " : " << int(hdr.msg_cnt) << " : " << hdr.len << " : " << sizeof...(msgs) <<endl;
    memcpy(buf, reinterpret_cast<void*>(&hdr), sizeof(SRPacketHeader));

    msg_seq_num += sizeof...(Msgs);

    return len;
}

#pragma pack(push,1)
struct SRDataTransport
{
    TransHeader     th;
    SRPacketHeader  ph;
    char            dt[MTULEN-sizeof(TransHeader)-sizeof(SRPacketHeader)];
};
#pragma pack(pop)

using namespace MKTDATASYSTEM::INPUTS;

struct StingRayReadInterface : RecvInterface
{
    StingRayReadInterface(const deque<const Line*>& lines_dq, const size_t proc) : RecvInterface(lines_dq, proc) {}

    bool init()
    {
         ///
         uint32_t seqnum{17};
         SRFeedTime fd_tm{171234};

         {
            SRDataTransport dt;
            dt.ph.len = buildPacket<45>(reinterpret_cast<char*>(&dt.ph), seqnum,
                         SRSecurityDefinition{fd_tm-1,'S',"ken",1968,10},
                         SRSecurityDefinition{fd_tm-2,'S',"wall",1969,100});

            msgQ_.push_back(dt);
         }

         {
            SRDataTransport dt;
            dt.ph.len = buildPacket<45>(reinterpret_cast<char*>(&dt.ph), seqnum,
                         SRAdd{fd_tm,'A',1717,68,100,1968,'B'},
                         SRAdd{fd_tm+1,'A',1718,67,100,1968,'B'});

            msgQ_.push_back(dt);
         }

         {
            SRDataTransport dt;
            dt.ph.len = buildPacket<45>(reinterpret_cast<char*>(&dt.ph), seqnum,
                         SRAdd{fd_tm+2,'A',1719,70,100,1968,'S'},
                         SRAdd{fd_tm+3,'A',1720,69,100,1968,'S'}, SRAdd{fd_tm+4,'A',1721,69,100,1968,'S'});

            msgQ_.push_back(dt);
         }

         {
            SRDataTransport dt;
            dt.ph.len = buildPacket<45>(reinterpret_cast<char*>(&dt.ph), seqnum,
                         SRAdd{fd_tm+4,'A',1722,56,100,1969,'S'},
                         SRAdd{fd_tm+5,'A',1723,55,100,1969,'S'}, SRAdd{fd_tm+6,'A',1724,55,100,1969,'B'});

            msgQ_.push_back(dt);
         }

         {
            SRDataTransport dt;
            dt.ph.len = buildPacket<45>(reinterpret_cast<char*>(&dt.ph), seqnum,
                         SRCan{fd_tm+7,'C',1719,30},
                         SRCan{fd_tm+8,'C',1724,33});

            msgQ_.push_back(dt);
         }

         {
            SRDataTransport dt;
            dt.ph.len = buildPacket<45>(reinterpret_cast<char*>(&dt.ph), seqnum,
                         SRDel{fd_tm+9,'D',1719},
                         SRDel{fd_tm+10,'D',1724});

            msgQ_.push_back(dt);
         }

         {
            SRDataTransport dt;
            dt.ph.len = buildPacket<45>(reinterpret_cast<char*>(&dt.ph), seqnum,
                         SRMod{fd_tm+11,'M',1723,85,56,true},
                         SRMod{fd_tm+12,'M',1723,76,57,true});

            msgQ_.push_back(dt);
         }

         {
            SRDataTransport dt;
            dt.ph.len = buildPacket<45>(reinterpret_cast<char*>(&dt.ph), seqnum,
                         SRRep{fd_tm+13,'R',1723,1725,75,56},
                         SRDel{fd_tm+14,'D',1725});

            msgQ_.push_back(dt);
         }
         return true;
    }

    bool join(const PhysLinePtr ppl)
    {
        ///
        return true;
    }

    bool read(const Line*& lp)
    {
        static size_t ridx{};

        if(ridx == msgQ_.size()) return false;
//cout << __FILE__ << " : "<< __LINE__ <<endl;
        auto& srdt = msgQ_[ridx++];

        /// in a real recv interface, a logical line obj would be available
       // static Line ln;

        /// the first Line has an arbitrated packet
        lp = lines_dq.front();

        /// the first physical line packet arrived first
      //  lp->pArb_line = lp->phys_lines_dq.front();

        Packet& pkt{currPkt_};//lp->pArb_line->ab_pkt};

        memcpy(const_cast<char*>(pkt.data), reinterpret_cast<const char*>(&srdt.dt), srdt.ph.len);
        pkt.len = srdt.ph.len;
        pkt.first_seq_num = srdt.ph.msg_seq_num;
        pkt.msg_cnt = srdt.ph.msg_cnt;

        lp->set_arb_pkt(&pkt);
        //lp->p_arb_pkt = &pkt;

        /// for test, should be done via arbitration code
       // lp->pArb_line->stats.incr(pkt.msg_cnt, pkt.len);

cout << __FILE__ << " : "<< __LINE__ << " : " << pkt.first_seq_num<< " : " << int(pkt.msg_cnt) << " : " << pkt.len <<endl;

        return true;
    }

private:
    std::deque<SRDataTransport> msgQ_;
    Packet currPkt_;
};

}

#endif // STINGRAYPUB_H_INCLUDED
