#ifndef MKTDATASYSTEMSTINGRAYMSGS_H_INCLUDED
#define MKTDATASYSTEMSTINGRAYMSGS_H_INCLUDED

#include "MktDataSystemBaseDefinitions.h"

///
/// this mkt data system will create it's own msgs that reflect the Sting Ray definitions
///

namespace MKTDATASYSTEM::FEEDHANDLER::STINGRAY
{
using namespace MKTDATASYSTEM::DEFINITIONS;

/**
using SROrderID = uint64_t;
using SRSize    = uint32_t;
using SRSymbolIdx = uint16_t;
using SRSymbol = char[8];
using SRPrice = int64_t;
using SRPriceScale = uint8_t;
using SRSide = char;//'B' or 'S'
using SRMsgType = char;
using SRFeedTime = uint64_t;

from MKTDATASYSTEM::DEFINITIONS:
using SymbolType = char[8];
using PriceType = int64_t;
using SizeType = uint32_t;
using SeqNumType = uint64_t;
using UniqueUpdateID = SeqNumType;
using DataLenType = uint16_t;
using OrderID = uint64_t;
using SymIdxType = uint32_t;
using FeedTimeType = uint64_t;
*/

/// not specific to SR
/*struct TransHeader
{
    char src[8];
    char dst[8];
    char ver[8];
    char pad[8];
};
*/
#pragma pack(push,1)

struct SRMsgPacketHeader
{
    uint32_t msg_seq_num;
    uint16_t len;
    uint8_t ver;
    uint8_t msg_cnt;
};

//template<char t>
struct SRMsgHeader
{
    FeedTimeType fd_tm;
    char type;
};

struct SRMsgSecurityDefinition : SRMsgHeader//<'S'>
{
    SymbolType       sym;
    uint16_t         idx;
    uint16_t         scale;
};

struct SRMsgAdd : SRMsgHeader//<'A'>
{
    OrderID   id;
    PriceType     prc;
    SizeType      sz;
    uint16_t sidx;
    char      sd;
};

struct SRMsgDel : SRMsgHeader//<'D'>
{
    OrderID   id;
};

struct SRMsgCan : SRMsgHeader//<'C'>
{
    OrderID   id;
    SizeType      sz;
};

struct SRMsgMod : SRMsgHeader//<'M'>
{
    OrderID   oid;
    SizeType      nsz;
    PriceType     nprc;
    bool        npriority;
};

struct SRMsgRep : SRMsgHeader//<'R'>
{
    OrderID   oid;
    OrderID   nid;
    SizeType      nsz;
    PriceType     nprc;
};

#pragma pack(pop)

}


#endif // MKTDATASYSTEMSTINGRAYMSGS_H_INCLUDED
