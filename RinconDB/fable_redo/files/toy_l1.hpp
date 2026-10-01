#pragma once
// toy_l1.hpp — stand-in L1 wire format with the shape of a SIP quote feed. Replace with the
// CTA (CQS) / UTP (UQDF) offsets, byte order and message types; also add one struct per
// additional feed with its own kProtoId. The store only needs the three members below.
//
// Packet : [u8 msg_count][u8 version][u16 rsvd][u32 seq] then msg_count messages
// Message: 0 u16 len | 2 u8 type ('Q' BBO,'N' NBBO) | 3 u8 exch | 4 u32 rsvd | 8 u64 feed_ns
//          16 char sym[11] | 27 rsvd[5] | 32 i64 bid_px | 40 i64 ask_px | 48 u32 bid_sz | 52 u32 ask_sz
//          'N' only: 56 u8 best_bid_exch | 57 u8 best_ask_exch

#include "rawmd.hpp"

namespace rawmd {

struct ToyL1 {
  static constexpr uint8_t  kProtoId = 1;
  static constexpr uint32_t kPktHdr = 8, kSymOff = 16, kSymLen = 11, kQuoteLen = 56, kNbboLen = 58;

  template <class F>
  [[gnu::always_inline]] static void for_each_msg(const uint8_t* p, std::size_t n, F&& f) noexcept {
    if (n < kPktHdr) [[unlikely]] return;
    const uint32_t cnt = p[0];
    std::size_t off = kPktHdr;
    for (uint32_t i = 0; i < cnt && off + 2 <= n; ++i) {
      uint16_t len;
      std::memcpy(&len, p + off, 2);
      if (len < kQuoteLen || off + len > n) [[unlikely]] return;
      uint64_t ts;
      std::memcpy(&ts, p + off + 8, 8);
      f(p + off, uint32_t{len}, p[off + 2], make_key(p + off + kSymOff, kSymLen), ts);
      off += len;
    }
  }

  static bool decode(const RecHdr& h, const uint8_t* m, OutQuote& o) noexcept {
    if (h.len < kQuoteLen) return false;
    std::memcpy(&o.feed_ns, m + 8, 8);
    o.rx_ns = h.rx_ns;
    std::memcpy(&o.bid_px, m + 32, 8);
    std::memcpy(&o.ask_px, m + 40, 8);
    std::memcpy(&o.bid_sz, m + 48, 4);
    std::memcpy(&o.ask_sz, m + 52, 4);
    const SymKey k = make_key(m + kSymOff, kSymLen);
    std::memcpy(o.sym, &k, 16);
    o.type = static_cast<char>(h.type);
    o.exch = static_cast<char>(m[3]);
    const bool nbbo = h.type == 'N' && h.len >= kNbboLen;
    o.bid_exch = nbbo ? static_cast<char>(m[56]) : ' ';
    o.ask_exch = nbbo ? static_cast<char>(m[57]) : ' ';
    o.proto = h.proto;
    o.rsvd[0] = o.rsvd[1] = o.rsvd[2] = 0;
    return true;
  }
};

}  // namespace rawmd
