#ifndef STINGRAYEXCH_H_INCLUDED
#define STINGRAYEXCH_H_INCLUDED

///
/// the hypothetical String Ray exchange defines their msgs as such
///

namespace STINGRAY
{
using SROrderID = uint64_t;
using SRSize    = uint32_t;
using SRSymbolIdx = uint16_t;
using SRSymbol = char[8];
using SRPrice = int64_t;
using SRPriceScale = uint16_t;
using SRSide = char;//'B' or 'S'
using SRMsgType = char;
using SRFeedTime = uint64_t;
using SRPriorityReset = bool;

/// not specific to SR
/// like a UDPIP Header
struct TransHeader
{
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    char ver[8];
    char pad[8];
};

#pragma pack(push,1)

/// typical content of an exchange packet header
struct SRPacketHeader
{
    uint32_t msg_seq_num;
    uint16_t len;/// length of msg content
    uint8_t ver;
    uint8_t msg_cnt;
};

//template<char t>
struct SRHeader
{
    SRFeedTime fd_tm;
    SRMsgType type;
};

struct SRSecurityDefinition : SRHeader//<'S'>
{
    SRSymbol        sym;
    SRSymbolIdx     idx;
    SRPriceScale    scale;
};

struct SRAdd : SRHeader//<'A'>
{
    SROrderID   id;
    SRPrice     prc;
    SRSize      sz;
    SRSymbolIdx sidx;
    SRSide      sd;
};

struct SRDel : SRHeader//<'D'>
{
    SROrderID   id;
};

struct SRCan : SRHeader//<'C'>
{
    SROrderID   id;
    SRSize      sz;
};

struct SRMod : SRHeader//<'M'>
{
    SROrderID   oid;
    SRSize      nsz;
    SRPrice     nprc;
    SRPriorityReset npriority;
};

struct SRRep : SRHeader//<'R'>
{
    SROrderID   oid;
    SROrderID   nid;
    SRSize      nsz;
    SRPrice     nprc;
};

#pragma pack(pop)
}

#endif // STINGRAYEXCH_H_INCLUDED
