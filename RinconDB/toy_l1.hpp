#pragma once
// toy_l1.hpp — stand-in L1 wire format with the shape of a SIP quote feed.
// Replace offsets / byte order / message types with the CTA (CQS) and UTP (UQDF) specs;
// the LineHandler and client path only depend on the two static functions below.
//
// Packet : [u8 msg_count][u8 version][u16 rsvd][u32 seq] then msg_count messages
// Message: off 0 u16 len | 2 u8 type ('Q' BBO, 'N' NBBO) | 3 u8 exch | 4 u32 rsvd
//          8 u64 sip_ns | 16 char sym[11] | 27 rsvd[5]
//          32 i64 bid_px | 40 i64 ask_px (x1e4) | 48 u32 bid_sz | 52 u32 ask_sz
//          'N' only: 56 u8 best_bid_exch | 57 u8 best_ask_exch

#include "rawmd.hpp"

namespace rawmd {

struct OutQuote {        // client wire format, 64 bytes, little-endian
  uint64_t rx_ns;
  uint64_t sip_ns;
  int64_t  bid_px;
  int64_t  ask_px;
  uint32_t bid_sz;
  uint32_t ask_sz;
  char     sym[16];
  char     type, exch, bid_exch, ask_exch;
  uint32_t rsvd;
};
static_assert(sizeof(OutQuote) == 64);

struct ToyL1 {
  static constexpr uint32_t kPktHdr = 8, kSymOff = 16, kSymLen = 11, kQuoteLen = 56, kNbboLen = 58;

  template <class F>
  [[gnu::always_inline]] static void for_each_msg(const uint8_t* p, std::size_t n, F&& f) noexcept {
    if (n < kPktHdr) [[unlikely]] return;
    const uint32_t cnt = p[0];
    std::size_t off = kPktHdr;
    for (uint32_t i = 0; i < cnt && off + 2 <= n; ++i) {
      uint16_t len;
      std::memcpy(&len, p + off, 2);
      if (len < kQuoteLen || off + len > n) [[unlikely]] return;  // malformed: stop this packet
      f(p + off, uint32_t{len}, p[off + 2], make_key(p + off + kSymOff, kSymLen));
      off += len;
    }
  }

  static bool decode(const uint8_t* m, uint32_t len, uint8_t type, uint64_t rx_ns, OutQuote& o) noexcept {
    if (len < kQuoteLen) return false;
    o.rx_ns = rx_ns;
    std::memcpy(&o.sip_ns, m + 8, 8);
    std::memcpy(&o.bid_px, m + 32, 8);
    std::memcpy(&o.ask_px, m + 40, 8);
    std::memcpy(&o.bid_sz, m + 48, 4);
    std::memcpy(&o.ask_sz, m + 52, 4);
    const SymKey k = make_key(m + kSymOff, kSymLen);
    std::memcpy(o.sym, &k, 16);
    o.type = static_cast<char>(type);
    o.exch = static_cast<char>(m[3]);
    const bool nbbo = type == 'N' && len >= kNbboLen;
    o.bid_exch = nbbo ? static_cast<char>(m[56]) : ' ';
    o.ask_exch = nbbo ? static_cast<char>(m[57]) : ' ';
    o.rsvd = 0;
    return true;
  }
};

}  // namespace rawmd
