#ifndef MKTDATASYSTEMSTINGRAYFEEDHANDLER_H_INCLUDED
#define MKTDATASYSTEMSTINGRAYFEEDHANDLER_H_INCLUDED

#include "MktDataSystemBaseDefinitions.h"
#include "MktDataSystemFeedHandler.h"
#include "MktDataSystemTicks.h"
#include "MktDataSystemStingRayMsgs.h"

namespace MKTDATASYSTEM::FEEDHANDLER
{
using namespace MKTDATASYSTEM::INPUTS;
using namespace MKTDATASYSTEM::DEFINITIONS;
using namespace MKTDATASYSTEM::FEEDHANDLER::STINGRAY;
namespace ticks = MKTDATASYSTEM::TICK;

template<Constants C, /** ???derived_from<TickProcessor>*/typename TP>
struct StingRayLineHandler : LineHandler
{
    static constexpr auto side_convert = [](const char sd)
    {
        if(sd =='B') return Side::Buy;
        else if (sd =='S') return Side::Sell;

        return Side::Unknown;
    };

    static constexpr auto get_uuid = [](){return UniqueUpdateID{};};

    StingRayLineHandler(const string& name, initializer_list<const Line*> il) : LineHandler(name, il) {}

    /// loop over packet processing it's contained msgs -- can do a
    bool process(const Line* lp) override;

    DataLenType add(const PktMsg& msg);
    DataLenType rep(const PktMsg& msg);
    DataLenType mod(const PktMsg& msg);
    DataLenType can(const PktMsg& msg);
    DataLenType del(const PktMsg& msg);
    DataLenType exec(const PktMsg& msg);
    DataLenType status(const PktMsg& msg);
    DataLenType security_def(const PktMsg& msg);

private:
    TP tkprocess;
};
/*
struct StingRayFeedHandler : FeedHandler<FeedFormat::STINGRAYBINARY, StingRayLineHandler>
{
    ///

};
*/
template<Constants C,/** ???derived_from<TickProcessor>*/typename TP>
bool StingRayLineHandler<C,TP>::process(const Line* lp)
{
    /// use Line info for deriving UUID, etc

 //   cout << __FILE__ << " : "<< __LINE__ << " : lp = " << lp <<endl;

    const Packet& pkt{*(lp->p_arb_pkt)};//pArb_line->ab_pkt};

    DataLenType offset{};
    for(uint8_t m = 0; m < pkt.msg_cnt; ++m)
    {
        PktMsg pm{pkt.data+offset, pkt.first_seq_num+m};
        const SRMsgHeader* phdr{reinterpret_cast<const SRMsgHeader*>(pm.data)};

        cout << __FILE__ << " : "<< __LINE__ << " : " << phdr->type <<endl;

        switch(phdr->type)
        {
        case 'A':
            offset += add(pm);
            break;

        case 'D':
            offset += del(pm);
            break;

        case 'C':
            offset += can(pm);
            break;

        case 'M':
            offset += mod(pm);
            break;

        case 'R':
            offset += rep(pm);
            break;

        case 'S':
            offset += security_def(pm);
            break;

        default:
            break;
        }
    }

    return true;
}

template<Constants C,/** ???derived_from<TickProcessor>*/typename TP>
DataLenType StingRayLineHandler<C,TP>::add(const PktMsg& pm)
{
    using localMsg = SRMsgAdd;

    const localMsg* padd{reinterpret_cast<const localMsg*>(pm.data)};

    ticks::Add<C.XID,C.FID> tk({nullptr, padd->sidx},
                           padd->id, padd->prc, padd->sz, side_convert(padd->sd), padd->fd_tm, get_uuid());

    tkprocess(tk);
//cout << __FILE__ << " : "<< __LINE__ <<endl;
    return sizeof(localMsg);
}

template<Constants C,/** ???derived_from<TickProcessor>*/typename TP>
DataLenType StingRayLineHandler<C,TP>::del(const PktMsg& pm)
{
    using localMsg = SRMsgDel;

    const localMsg* pmsg{reinterpret_cast<const localMsg*>(pm.data)};

    ticks::Del<C.XID,C.FID> tk(pmsg->id, pmsg->fd_tm, get_uuid());

    tkprocess(tk);
//cout << __FILE__ << " : "<< __LINE__ <<endl;
    return sizeof(localMsg);
}

template<Constants C,/** ???derived_from<TickProcessor>*/typename TP>
DataLenType StingRayLineHandler<C,TP>::can(const PktMsg& pm)
{
    using localMsg = SRMsgCan;

    const localMsg* pmsg{reinterpret_cast<const localMsg*>(pm.data)};

    ticks::Can<C.XID,C.FID> tk(pmsg->id, pmsg->sz, pmsg->fd_tm, get_uuid());

    tkprocess(tk);
//cout << __FILE__ << " : "<< __LINE__ <<endl;
    return sizeof(localMsg);
}

template<Constants C,/** ???derived_from<TickProcessor>*/typename TP>
DataLenType StingRayLineHandler<C,TP>::rep(const PktMsg& pm)
{
    using localMsg = SRMsgRep;

    const localMsg* pmsg{reinterpret_cast<const localMsg*>(pm.data)};

    ticks::Rep<C.XID,C.FID> tk(pmsg->oid, pmsg->nid, pmsg->nprc, pmsg->nsz, pmsg->fd_tm, get_uuid());

    tkprocess(tk);
//cout << __FILE__ << " : "<< __LINE__ <<endl;
    return sizeof(localMsg);
}

template<Constants C,/** ???derived_from<TickProcessor>*/typename TP>
DataLenType StingRayLineHandler<C,TP>::mod(const PktMsg& pm)
{
    using localMsg = SRMsgMod;

    const localMsg* pmsg{reinterpret_cast<const localMsg*>(pm.data)};

    ticks::Mod<C.XID,C.FID> tk(pmsg->oid, pmsg->nprc, pmsg->nsz,
                               (pmsg->npriority == true) ? OrderPriority::Reset : OrderPriority::Maintain, pmsg->fd_tm, get_uuid());

    tkprocess(tk);
//cout << __FILE__ << " : "<< __LINE__ <<endl;
    return sizeof(localMsg);
}

template<Constants C,/** ???derived_from<TickProcessor>*/typename TP>
DataLenType StingRayLineHandler<C,TP>::security_def(const PktMsg& pm)
{
    using localMsg = SRMsgSecurityDefinition;

    const localMsg* pmsg{reinterpret_cast<const localMsg*>(pm.data)};

    ticks::Symbol<C.XID,C.FID> tk(pmsg->sym, pmsg->idx, pmsg->scale, pmsg->fd_tm, get_uuid());

    tkprocess(tk);
//cout << __FILE__ << " : "<< __LINE__ << " pss = " << pss <<endl;
    return sizeof(localMsg);
}

}

#endif // MKTDATASYSTEMSTINGRAYFEEDHANDLER_H_INCLUDED
