#ifndef MKTDATASYSTEMTESTING_H_INCLUDED
#define MKTDATASYSTEMTESTING_H_INCLUDED

#include "MktDataSystemBaseDefinitions.h"
//#include "MktDataSystemInputs.h"
#include "StingRayExch.h"

namespace EXAMPLEINPUTS
{
//using namespace MKTDATASYSTEM::INPUTS;
//constexpr size_t MTULEN{1500};
using namespace STINGRAY;
using namespace MKTDATASYSTEM::DEFINITIONS;

#pragma pack(push,1)

/// example - not same as real
struct RawPacketIPHdr
{
    uint32_t version_and_stuff{};
    uint32_t checksum_and_stuff{};
    uint32_t src_ip{};
    uint32_t dst_ip{};
};
static_assert(sizeof(RawPacketIPHdr) == 16);

struct RawUDPIPPacketHdr : RawPacketIPHdr
{
    uint16_t src_port{};
    uint16_t dst_port{};
    uint16_t dlen{};///length of exch data portion
    uint16_t checksum{};
};
//    char     data[MTULEN-sizeof(RawPacketIPHdr) - 8];
//};
static_assert(sizeof(RawUDPIPPacketHdr) == 24);

///use this for storage in a container that needs fixed size elements (eg, STL deque);
template<typename EXCHHEADER = SRPacketHeader>
struct FakeRawFullPacket
{
    RawUDPIPPacketHdr   ntwhdr;
    EXCHHEADER          exchhdr;
    char                exchdata[MTULEN-sizeof(RawUDPIPPacketHdr) - sizeof(EXCHHEADER)];
    constexpr static size_t data_offset{sizeof(RawUDPIPPacketHdr)+sizeof(EXCHHEADER)};
};

#pragma pack(pop)

template<typename EXCHHEADER>
Packet convertPkt(const FakeRawFullPacket<EXCHHEADER>& rawPkt)
{
    if constexpr(is_same_v<EXCHHEADER,SRPacketHeader>)
        return {{rawPkt.ntwhdr.src_ip,rawPkt.ntwhdr.src_port}, rawPkt.exchhdr.len,
                rawPkt.exchhdr.msg_seq_num, rawPkt.exchhdr.msg_cnt,
                /*reinterpret_cast<const char*>(*/&rawPkt.exchdata[0]};
    else
        return Packet();
}
/*
struct SRPacketHeader
{
    uint32_t msg_seq_num;
    uint16_t len;/// length of msg content
    uint8_t ver;
    uint8_t msg_cnt;
};

struct Packet
{
    struct Msg
    {
        const char* data{};
    //    const FeedTimeType feed_time;
        //DataLenType len{};
        SeqNumType seq_num;
    };

    LineHdr line;

    DataLenType len{};
    SeqNumType first_seq_num;
    uint8_t msg_cnt;
    char* data;//[2000];
    bool has_been_analyzed{};
};
*/

using namespace STINGRAY;

template<typename EXCHHEADER, uint8_t ver>//, typename...Msgs>
constexpr uint16_t buildRawUDPIPPacket(char* buf, uint32_t& msg_seq_num,
                             const RawUDPIPPacketHdr& rawHdr,
                            auto&&... msgs)
{
    uint16_t offset{sizeof(RawUDPIPPacketHdr) + sizeof(EXCHHEADER)};
    uint16_t  len{};/// length of exch msgs

    static auto pack = [&buf, offset, &len](auto&& msg)
    {
        memcpy(buf+offset+len, reinterpret_cast<void*>(&msg), sizeof(msg));
        len += sizeof(msg);
    };

    (..., pack(msgs));

    EXCHHEADER hdr{msg_seq_num, len, ver, sizeof...(msgs)};

    const_cast<uint16_t&>(rawHdr.dlen) = sizeof(EXCHHEADER) + len;/// exch data length
//cout << __FILE__ << " : "<< __LINE__ << " : " << hdr.msg_seq_num<< " : "
//    << int(hdr.msg_cnt) << " : " << hdr.len << " : " << sizeof...(msgs) << " : "
//    << rawHdr.dlen<<endl;

    memcpy(buf, reinterpret_cast<const char*>(&rawHdr), sizeof(RawUDPIPPacketHdr));
    memcpy(buf+sizeof(RawUDPIPPacketHdr), reinterpret_cast<char*>(&hdr), sizeof(EXCHHEADER));

    msg_seq_num += sizeof...(msgs);

    return len;
}

//3/17/26
//need to associate msg_seq_num directly with a specific physical channel


template<typename EXCHHEADER, uint8_t ver>
struct RawPacketBuilder
{
 //   RawPacketBuilder(const uint32_t first_msg_seq_num = 0)
 //       : msg_seq_num(first_msg_seq_num){}

    template<typename... Msgs>
    void add(const LineHdr& srcMCGroup, const LineHdr& dstMCGroup, uint32_t& msg_seq_num,
             Msgs&&... msgs)
    {
        FakeRawFullPacket<EXCHHEADER> pkt;
        buildRawUDPIPPacket<EXCHHEADER,ver>(reinterpret_cast<char*>(&pkt), msg_seq_num,
                            RawUDPIPPacketHdr({ver,0,srcMCGroup.mc_ip, dstMCGroup.mc_ip,
                             srcMCGroup.mc_port, dstMCGroup.mc_port,0,0}),
                            forward<Msgs>(msgs)...);

//        cout << __FILE__ << " : "<< __LINE__ << " after buildRawUDPIPPacket " << endl;
        pktstream_.emplace_back(pkt);
    }

    deque<FakeRawFullPacket<EXCHHEADER>>& getPktStream(){return pktstream_;}

private:
    deque<FakeRawFullPacket<EXCHHEADER>> pktstream_;
  //  uint32_t msg_seq_num;
    uint64_t fd_tm{171234};
};

template<uint8_t VERSION, size_t NUM_PROCESSORS>
struct MultiLineStingRayTest
{
    static MultiLineStingRayTest& inst()
    {
        static MultiLineStingRayTest obj;
        return obj;
    }

    deque<FakeRawFullPacket<SRPacketHeader>>& getProcPktStream(const size_t proc){return pktbldr_[proc].getPktStream();}
private:

    MultiLineStingRayTest(){build();}

    MultiLineStingRayTest(const MultiLineStingRayTest&) = delete;
    MultiLineStingRayTest& operator=(const MultiLineStingRayTest&) = delete;

    void build();

    RawPacketBuilder<SRPacketHeader,VERSION> pktbldr_[NUM_PROCESSORS];
};

template<uint8_t VERSION, size_t NUM_PROCESSORS>
void MultiLineStingRayTest<VERSION, NUM_PROCESSORS>::build()
{
    ///**********************************************************
    /// setup logical and physical line definitions
    ///  * simplify access to these
    struct PhysLineData
    {
        const LineHdr hdr;
        uint32_t msg_seq_num;
    };
    constexpr size_t NUM_PHYS_LINES{8};

    PhysLineData phys_linedata_arr[NUM_PHYS_LINES] =
    {
        {{inet_addr_constexpr("224.0.59.0"), 23200}, 17},
        {{inet_addr_constexpr("224.0.59.128"), 23700},17},
        {{inet_addr_constexpr("224.0.59.1"), 23201}, 17},
        {{inet_addr_constexpr("224.0.59.129"), 23701}, 17},
        {{inet_addr_constexpr("224.0.59.2"), 23202}, 17},
        {{inet_addr_constexpr("224.0.59.130"), 23702}, 17},
        {{inet_addr_constexpr("224.0.59.3"), 23203}, 17},
        {{inet_addr_constexpr("224.0.59.131"), 23703}, 17}
    };

    struct LogicalLineData
    {
        const size_t A;
        const size_t B;
        uint32_t expected_msg_seq_num;
    };

    constexpr size_t NUM_LOGICAL_LINES{4};

    LogicalLineData logical_linedata_arr[NUM_LOGICAL_LINES] =
    {
        {0,1,17}, {2,3,17}, {4,5,17}, {6,7,17}
    };

    /// simplify access
    auto& ln1_A{phys_linedata_arr[logical_linedata_arr[0].A]};
    auto& ln1_B{phys_linedata_arr[logical_linedata_arr[0].B]};
    auto& ln2_A{phys_linedata_arr[logical_linedata_arr[1].A]};
    auto& ln2_B{phys_linedata_arr[logical_linedata_arr[1].B]};
    auto& ln3_A{phys_linedata_arr[logical_linedata_arr[2].A]};
    auto& ln3_B{phys_linedata_arr[logical_linedata_arr[2].B]};
    auto& ln4_A{phys_linedata_arr[logical_linedata_arr[3].A]};
    auto& ln4_B{phys_linedata_arr[logical_linedata_arr[3].B]};

    /// line processors -- aggregate of multiple lines
    auto& lp1_pktbldr{pktbldr_[0]};
    auto& lp2_pktbldr{pktbldr_[1]};

    ///**********************************************************

    SRFeedTime fd_tm{171234};

    /// A,B should get the same security definition msgs
    lp1_pktbldr.add(ln1_A.hdr, {0,0}, ln1_A.msg_seq_num,
                 SRSecurityDefinition{fd_tm-1,'S',"ken",1968,10},
                         SRSecurityDefinition{fd_tm-2,'S',"wall",1969,100});

//cout << __FILE__ << " : "<< __LINE__ << " after add first msg " << endl;

    lp1_pktbldr.add(ln1_B.hdr, {0,0}, ln1_B.msg_seq_num,
                 SRSecurityDefinition{fd_tm-1,'S',"ken",1968,10},
                         SRSecurityDefinition{fd_tm-2,'S',"wall",1969,100});


    lp1_pktbldr.add(ln1_A.hdr, {0,0}, ln1_A.msg_seq_num,
                 SRAdd{fd_tm,'A',1717,68,100,1968,'B'},
                 SRAdd{fd_tm+1,'A',1718,67,100,1968,'B'});

}

}

#endif // MKTDATASYSTEMTESTING_H_INCLUDED
