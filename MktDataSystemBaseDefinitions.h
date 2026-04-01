#ifndef MKTDATASYSTEMBASEDEFINITIONS_H_INCLUDED
#define MKTDATASYSTEMBASEDEFINITIONS_H_INCLUDED

namespace MKTDATASYSTEM::DEFINITIONS
{
using SymbolType = char[8];
using PriceType = int64_t;
using SizeType = uint32_t;
using SeqNumType = uint64_t;
using UniqueUpdateID = SeqNumType;
using DataLenType = uint16_t;
using OrderID = uint64_t;
using SymIdxType = uint64_t;
using PriceScaleType = uint16_t;
using TimeType = uint64_t;
using FeedTimeType = TimeType;
using NumOrdersType = uint32_t;/// ??

inline constexpr SymIdxType DefaultSymIdx = numeric_limits<SymIdxType>::max();
inline constexpr PriceType DefaultPrice = numeric_limits<PriceType>::max();
inline constexpr SizeType DefaultSize = numeric_limits<SizeType>::max();
inline constexpr SizeType DefaultOrderID = numeric_limits<OrderID>::max();

inline constexpr uint8_t MAX_PUB_LEVELS_TMP{10};

enum class Side : uint8_t{Unknown, Buy, Sell};

struct PrcLevelGeneric
{
    PriceType prc{};
    SizeType agg_sz{};/// aggregate size
    NumOrdersType num_orders{};
    FeedTimeType feed_time{};
    UniqueUpdateID uuid{};

    PrcLevelGeneric(const PriceType prc, const SizeType agg_sz, const NumOrdersType num_orders,
                    const FeedTimeType feed_time, const UniqueUpdateID uuid) noexcept : prc(prc), agg_sz(agg_sz),
                    num_orders(num_orders), feed_time(feed_time), uuid(uuid){}

    PrcLevelGeneric() noexcept{}
};

static_assert(sizeof(PrcLevelGeneric) == 32);

enum class MsgType : uint8_t
{
    Unknown = 0,
    Add, Cancel, Modify, Replace, Delete, Execute, Trade, Symbol, ExchStatus, SymbolStatus, OneSidedBookUpdate
};

template<typename H>
concept IsMsgType = is_same_v<H,MsgType>;

enum class OrderPriority : uint8_t
{
    Maintain = 0,
    Reset
};

enum class ExchID : uint8_t
{
    Unknown = 0,
    NYSE, NYSEARCA, NYSENSX, NYSETEXAS, NYSEAMER,
    NASDAQ, BX, PSX,
    BATS, BATSY, EDGE, EDGX,
    MEMX, IEX, LTSE, MIAX,
    STINGRAY
};

enum class FeedFormat : uint8_t
{
    Unknown = 0,
    ITCH5, NYSEINTEGRATED,STINGRAYBINARY
};


constexpr void set_sym(const char* s, SymbolType sym) noexcept
{
    if( s == nullptr)
    {
        sym[0] = '\0';

       // throw; // it's ok to create a SymbolObj with s == nullptr
       return;
    }

    const int len {strlen(s)};

    const int L{(len < sizeof(SymbolType)) ? len : sizeof(SymbolType) -1};
    strncpy(&sym[0], s, L);
    sym[L] = '\0';
}

struct SymbolObj
{
    constexpr SymbolObj(const char* s, const SymIdxType idx, const PriceScaleType scale = 1) noexcept : idx(idx), scale(scale) {set_sym(s, sym);}

    SymbolType sym{};
    const SymIdxType idx;
    const PriceScaleType scale;
};


struct SymbolState
{
    constexpr SymbolState(const SymbolObj& so) noexcept : sym_obj(so) { if(sym_obj.sym[0] == '\0') throw;}
    constexpr SymbolState(const char* s, const SymIdxType idx, const PriceScaleType scale = 1) noexcept :
                        sym_obj(s, idx, scale){if(sym_obj.sym[0] == '\0') throw;}

    SymbolObj sym_obj;

    SymIdxType id() const noexcept {return sym_obj.idx;}
    const char* symbol() const noexcept {return &sym_obj.sym[0];}
    PriceScaleType prc_scale() const noexcept {return sym_obj.scale;}
};

struct MemPoolInputs
{
    size_t Init;
    size_t GrowthFactor;/// factor =2 --> grow by 1/2
};

struct FastHashMapInputs
{
    uint16_t POWOF2_buckets;
    size_t INITVALCNT_bucket;
    MemPoolInputs MemPoolBlocks;
};

struct Constants
{
    ExchID XID;
    FeedFormat FID;

    inline static FastHashMapInputs fastOIDMapInputs{21,5, 1<<21,2};
    inline static FastHashMapInputs fastSymStateMapInputs{14,5, 1<<14,2};

    /// for the allocation of PrcLvls in the agg books
    inline static MemPoolInputs symStatePrcLvl{24*1000, 2};/// per TickProcessor (TP) --> shared among lines of TP
};

static_assert(default_initializable<Constants>);/// Constants must be a constexpr so as to be used as a template param

// A constexpr lambda to parse IPv4 strings at compile-time
constexpr auto inet_addr_constexpr = [](const char* cp) -> uint32_t {
    uint32_t val = 0;
    uint32_t part = 0;
    int dots = 0;

    while (*cp) {
        if (*cp >= '0' && *cp <= '9') {
            part = part * 10 + (*cp - '0');
        } else if (*cp == '.') {
            val = (val << 8) | (part & 0xFF);
            part = 0;
            dots++;
        }
        cp++;
    }
    // Add the final octet
    val = (val << 8) | (part & 0xFF);

    // Note: Standard inet_addr returns network byte order (Big Endian)
    // This simple version may need __builtin_bswap32 for exact parity.
if constexpr (std::endian::native == std::endian::little)
    return std::byteswap(val);
else
    return val;

};

// Usage at compile-time
static_assert(inet_addr_constexpr("127.0.0.1") == 0x0100007F);

struct LineHdr
{
    uint32_t mc_ip{};/// use inet_addr(<IP string>) to set this
    uint16_t mc_port{};

    constexpr LineHdr() = default;
    constexpr LineHdr(const uint32_t ip, const uint16_t port) : mc_ip(ip), mc_port(port){}

    constexpr LineHdr(const LineHdr&) = default;
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
    const char* data;//[2000];
    bool has_been_analyzed{};
};

using PktMsg = Packet::Msg;

constexpr size_t max_line_name_sz{7};

constexpr void set_line_name(const char* s, char* lnm) noexcept
{
 /*   if( s == nullptr)
    {
        lnm[0] = '\0';

       // throw; // it's ok to create a SymbolObj with s == nullptr
       return;
    }
*/
    const int len {strlen(s)};

    const int L{(len <= max_line_name_sz) ? len : max_line_name_sz };
    //strncpy(lnm, s, L);
    for(size_t i = 0; i < L; ++i)
        lnm[i] = s[i];

    lnm[L] = '\0';
}


}

#endif // MKTDATASYSTEMBASEDEFINITIONS_H_INCLUDED
