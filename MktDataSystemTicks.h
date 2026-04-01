#ifndef MKTDATASYSTEMTICKS_H_INCLUDED
#define MKTDATASYSTEMTICKS_H_INCLUDED

#include "MktDataSystemBaseDefinitions.h"


/// ticks -- internal and external messages
///         * internal msgs are used between decoders and book building
///         * each msg has a header which includes symbol, exch, sequ num, msg type
///
/// wire -- the tick msg data classes
///

namespace MKTDATASYSTEM::WIRE
{
using namespace MKTDATASYSTEM::DEFINITIONS;

struct Header
{
    SymbolType      sym;
    ExchID          exch_id;
    FeedFormat      feed_id;
    MsgType         msg_id;
    char            pad1[5];
    SeqNumType      seq_num;/// published msg seq num
    FeedTimeType    feed_time;
    UniqueUpdateID  uuid;

    Header(const ExchID xid, const FeedFormat fid, const MsgType mid,
           const FeedTimeType ft, const UniqueUpdateID uid) : exch_id(xid),
        feed_id(fid), msg_id(mid), feed_time(ft), uuid(uid)
    {
    }
};

static_assert(sizeof(Header) == 40);



/// new order
struct Add : Header
{
    OrderID     oid;
    PriceType   prc;
    SizeType    sz;
    Side        side;

    Add(const ExchID xid, const FeedFormat fid, const FeedTimeType ft, const UniqueUpdateID uid,
        const OrderID oid, const PriceType prc, const SizeType sz,
        const Side sd) :
        Header(xid, fid, MsgType::Add, ft, uid),
        oid(oid), prc(prc), sz(sz), side(sd)
    {
    }
};

static_assert(sizeof(MKTDATASYSTEM::WIRE::Add) == 64);

/// full cancel of order
struct Del : Header
{
    OrderID     oid;

    Del(const ExchID xid, const FeedFormat fid, const FeedTimeType ft, const UniqueUpdateID uid,
        const OrderID oid) :
        Header(xid, fid, MsgType::Delete, ft, uid),
        oid(oid)
    {
    }
};

static_assert(sizeof(MKTDATASYSTEM::WIRE::Del) == 48);


/// partial cancel of order -- same order ID
struct Can : Header
{
    OrderID     oid;
    SizeType    sz; /// cancel size

    Can(const ExchID xid, const FeedFormat fid, const FeedTimeType ft, const UniqueUpdateID uid,
        const OrderID oid, const SizeType sz) :
        Header(xid, fid, MsgType::Cancel, ft, uid),
        oid(oid), sz(sz)
    {
    }
};

static_assert(sizeof(MKTDATASYSTEM::WIRE::Can) == 56);

struct Mod : Header
{
    OrderID     orig_oid;
    PriceType   new_prc;
    SizeType    new_sz; ///
    OrderPriority mod_priority;

    Mod(const ExchID xid, const FeedFormat fid, const FeedTimeType ft, const UniqueUpdateID uid,
        const OrderID ooid, const PriceType nprc, const SizeType nsz, const OrderPriority opri) :
        Header(xid, fid, MsgType::Modify, ft, uid),
        orig_oid(ooid), new_prc(nprc), new_sz(nsz), mod_priority(opri)
    {
    }
};

static_assert(sizeof(MKTDATASYSTEM::WIRE::Mod) == 64);

struct Rep : Header
{
    OrderID     orig_oid;
    OrderID     new_oid;
    PriceType   new_prc;
    SizeType    new_sz; ///

    Rep(const ExchID xid, const FeedFormat fid, const FeedTimeType ft, const UniqueUpdateID uid,
        const OrderID ooid, const OrderID noid,
            const PriceType nprc, const SizeType nsz) :
        Header(xid, fid, MsgType::Replace, ft, uid),
        orig_oid(ooid), new_oid(noid), new_prc(nprc), new_sz(nsz)
    {
    }
};

static_assert(sizeof(MKTDATASYSTEM::WIRE::Rep) == 72);

struct Exec : Header
{
    OrderID     oid;
    SizeType    sz; /// executed size

    Exec(const ExchID xid, const FeedFormat fid, const FeedTimeType ft, const UniqueUpdateID uid,
         const OrderID oid, const SizeType xsz) :
        Header(xid, fid, MsgType::Execute, ft, uid), oid(oid), sz(xsz)
    {
    }
};

static_assert(sizeof(MKTDATASYSTEM::WIRE::Exec) == 56);

/// good for publishing agg book updates for order based feeds - one order at a time publish
struct QuoteLevelsOneSide : Header
{
    QuoteLevelsOneSide(const ExchID xid, const FeedFormat fid, const FeedTimeType ft, const UniqueUpdateID uid,
                       const PriceScaleType prc_scale, const Side sd,
                       const uint8_t lvl, const uint8_t num) : Header(xid, fid, MsgType::OneSidedBookUpdate, ft, uid),
                       prc_scale(prc_scale), sd(sd), start_lvl(lvl), num(num)
                       {
                       }

    PriceScaleType prc_scale;
    Side sd;
    uint8_t start_lvl;
    uint8_t num;
    uint8_t pad1[1];
    PrcLevelGeneric lvls[MAX_PUB_LEVELS_TMP];/// ??? make more generic via template arg for num or just using payload concept
};

struct Symbol : Header
{
    Symbol(const ExchID xid, const FeedFormat fid, const FeedTimeType ft, const UniqueUpdateID uid,
            const SymIdxType sym_idx, const PriceScaleType prc_scale) :
                    Header(xid, fid, MsgType::Symbol, ft, uid), sym_idx(sym_idx), prc_scale(prc_scale)
                       {
                       }

    SymIdxType sym_idx;
    PriceScaleType prc_scale;
};

}

namespace MKTDATASYSTEM::TICK
{
using namespace MKTDATASYSTEM::DEFINITIONS;

struct Header : MKTDATASYSTEM::WIRE::Header
{
    void set_seq_num(const SeqNumType v) {seq_num = v;}
    void set(const ExchID id) {exch_id = id;}
    void set(const FeedFormat id) {feed_id = id;}
};

template<typename S>
concept IsSymType = (integral<remove_cvref_t<S>> || is_same_v<remove_cvref_t<S>,char*>);

template<ExchID XID, FeedFormat FID>//, IsSymType SYMTYPE>
struct Add : MKTDATASYSTEM::WIRE::Add
{
    Add(const SymbolObj& symob/*const char* s*//*MKTDATASYSTEM::DEFINITIONS::SymbolState* pss*/, const OrderID oid,
        const PriceType prc, const SizeType sz, const Side sd,
        const FeedTimeType ft, const UniqueUpdateID uid) :
            MKTDATASYSTEM::WIRE::Add(XID, FID, ft, uid, oid, prc, sz, sd), symob(symob)//, pss(pss)
    {
      //  MKTDATASYSTEM::set_sym(s, sym);
    }

    void set_sym(const char* s){set_sym(s, sym);}
    SymbolObj symob;
   // MKTDATASYSTEM::DEFINITIONS::SymbolState* pss;
};

template<ExchID XID, FeedFormat FID>
struct Del : MKTDATASYSTEM::WIRE::Del
{
    Del(/*const char* s,*/ const OrderID oid, const FeedTimeType ft, const UniqueUpdateID uid) :
        MKTDATASYSTEM::WIRE::Del(XID, FID, ft, uid, oid)
    {
      //  MKTDATASYSTEM::WIRE::set_sym(s, strlen(s), sym);
    }
};

template<ExchID XID, FeedFormat FID>
struct Can : MKTDATASYSTEM::WIRE::Can
{
    Can(/*const char* s,*/ const OrderID oid, const SizeType csz, const FeedTimeType ft, const UniqueUpdateID uid) :
        MKTDATASYSTEM::WIRE::Can(XID, FID, ft, uid, oid, csz)
    {
      //  MKTDATASYSTEM::WIRE::set_sym(s, strlen(s), sym);
    }
};

template<ExchID XID, FeedFormat FID>
struct Mod : MKTDATASYSTEM::WIRE::Mod
{
    Mod(const OrderID oid, const PriceType nprc,
        const SizeType nsz, const OrderPriority opri, const FeedTimeType ft, const UniqueUpdateID uid) :
        MKTDATASYSTEM::WIRE::Mod(XID, FID, ft, uid, oid, nprc, nsz, opri)
    {
      //  MKTDATASYSTEM::WIRE::set_sym(s, strlen(s), sym);
    }
};

template<ExchID XID, FeedFormat FID>
struct Rep : MKTDATASYSTEM::WIRE::Rep
{
    Rep(/*const char* s,*/ const OrderID ooid, const OrderID noid, const PriceType nprc,
        const SizeType nsz, const FeedTimeType ft, const UniqueUpdateID uid) :
        MKTDATASYSTEM::WIRE::Rep(XID, FID, ft, uid, ooid, noid, nprc, nsz)
    {
      //  MKTDATASYSTEM::WIRE::set_sym(s, strlen(s), sym);
    }
};

template<ExchID XID, FeedFormat FID>
struct Exec : MKTDATASYSTEM::WIRE::Exec
{
    Exec(/*const char* s,*/ const OrderID oid, const SizeType xsz, const FeedTimeType ft, const UniqueUpdateID uid) :
        MKTDATASYSTEM::WIRE::Exec(XID, FID, ft, uid, oid, xsz)
    {
      //  MKTDATASYSTEM::WIRE::set_sym(s, strlen(s), sym);
    }
};

template<template<ExchID, FeedFormat>typename T, ExchID XID, FeedFormat FID>
concept IsL3Tick = (same_as<T<XID,FID>, Add<XID,FID>> || same_as<T<XID,FID>, Del<XID,FID>> || same_as<T<XID,FID>, Can<XID,FID>>
                    || same_as<T<XID,FID>, Rep<XID,FID>> || same_as<T<XID,FID>, Exec<XID,FID>>);


template<ExchID XID, FeedFormat FID>
struct QuoteLevelsOneSide : MKTDATASYSTEM::WIRE::QuoteLevelsOneSide
{
    QuoteLevelsOneSide(const char* s, const PriceScaleType prc_scale, const Side sd, const uint8_t lvl, const uint8_t num,
                       const FeedTimeType ft, const UniqueUpdateID uid) :
        MKTDATASYSTEM::WIRE::QuoteLevelsOneSide(XID, FID, ft, uid, prc_scale, sd, lvl, num)
    {
        set_sym(s, sym);
    }
};

template<ExchID XID, FeedFormat FID>
struct Symbol : MKTDATASYSTEM::WIRE::Symbol
{
    Symbol(const char* s, const SymIdxType sym_idx, const PriceScaleType prc_scale, const FeedTimeType ft, const UniqueUpdateID uid) :
            MKTDATASYSTEM::WIRE::Symbol(XID, FID, ft, uid, sym_idx, prc_scale)
    {
        set_sym(s, sym);
    }
};

}
#endif // MKTDATASYSTEMTICKS_H_INCLUDED
