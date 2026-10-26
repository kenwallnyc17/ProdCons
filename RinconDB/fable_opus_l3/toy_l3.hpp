#pragma once
// toy_l3.hpp — stand-in L3 wire format with the shape of TotalView-ITCH (locate code in
// every message; symbol only in the directory and add messages). Little-endian here for
// the test; a real ITCH adapter byte-swaps in for_each_msg/decode and nothing else changes.
//
// Packet : [u8 msg_count][u8 version][u16 rsvd][u32 seq] then msg_count messages
// Header : 0 u16 len | 2 u8 type | 3 pad | 4 u16 locate | 6 u16 tracking | 8 u64 feed_ns
//   'R' directory : 16 sym[8]                                               (24)
//   'A' add       : 16 u64 ref | 24 u8 side | 28 u32 shares | 32 sym[8] | 40 u32 px  (44)
//   'E' executed  : 16 u64 ref | 24 u32 shares | 28 u64 match               (36)
//   'X' cancel    : 16 u64 ref | 24 u32 shares                              (28)
//   'D' delete    : 16 u64 ref                                              (24)
//   'U' replace   : 16 u64 ref | 24 u64 new_ref | 32 u32 shares | 36 u32 px (40)
//   'M' modify    : 16 u64 ref | 24 u32 shares | 28 u32 px                (32, PITCH-style, same id)
//   'P' trade     : 16 u64 ref | 24 u8 side | 28 u32 shares | 32 sym[8] | 40 u32 px  (44, no book effect)
//   'S' system    : locate 0, 16 u8 event                                   (17, not stored)

#include "l3.hpp"

namespace rawmd::l3 {

struct ToyL3 {
  static constexpr uint8_t  kProtoId   = 2;
  static constexpr bool     kHasLocate = true;
  static constexpr int8_t   kPxExp     = -4;
  static constexpr uint32_t kPktHdr = 8, kHdr = 16, kMaxMsg = 44;

  template <class T> static T rd(const uint8_t* p) noexcept { T v; std::memcpy(&v, p, sizeof v); return v; }

  template <class F>
  [[gnu::always_inline]] static void for_each_msg(const uint8_t* p, std::size_t n, F&& f) noexcept {
    if (n < kPktHdr) [[unlikely]] return;
    const uint32_t cnt = p[0];
    std::size_t off = kPktHdr;
    for (uint32_t i = 0; i < cnt && off + kHdr <= n; ++i) {
      const uint8_t* m = p + off;
      const uint16_t len = rd<uint16_t>(m);
      if (len < kHdr || off + len > n) [[unlikely]] return;
      MsgRef r;
      r.locate = rd<uint16_t>(m + 4);
      switch (m[2]) {
        case 'R': r.op = Op::Dir; r.key = make_key(m + 16, 8); break;
        case 'A': r.op = Op::Add; r.order_id = rd<uint64_t>(m + 16); r.qty = rd<uint32_t>(m + 28); r.key = make_key(m + 32, 8); break;
        case 'E': case 'X': r.op = Op::Reduce; r.order_id = rd<uint64_t>(m + 16); r.qty = rd<uint32_t>(m + 24); break;
        case 'D': r.op = Op::End; r.order_id = rd<uint64_t>(m + 16); break;
        case 'U': r.op = Op::Replace; r.order_id = rd<uint64_t>(m + 16); r.new_order_id = rd<uint64_t>(m + 24); r.qty = rd<uint32_t>(m + 32); break;
        case 'M': r.op = Op::Modify; r.order_id = rd<uint64_t>(m + 16); r.qty = rd<uint32_t>(m + 24); break;
        case 'P': r.op = Op::Sym; r.key = make_key(m + 32, 8); break;
        default:  r.op = Op::Skip; break;
      }
      f(m, uint32_t{len}, m[2], rd<uint64_t>(m + 8), r);
      off += len;
    }
  }

  static bool decode(const RecHdr& h, const uint8_t* m, BookEvent& e) noexcept {
    switch (h.type) {
      case 'A': e = {Ev::Add, m[24], rd<uint32_t>(m + 28), rd<uint32_t>(m + 40), rd<uint64_t>(m + 16), 0}; return true;
      case 'E': case 'X': e = {Ev::Reduce, 0, rd<uint32_t>(m + 24), 0, rd<uint64_t>(m + 16), 0}; return true;
      case 'D': e = {Ev::Delete, 0, 0, 0, rd<uint64_t>(m + 16), 0}; return true;
      case 'U': e = {Ev::Replace, 0, rd<uint32_t>(m + 32), rd<uint32_t>(m + 36), rd<uint64_t>(m + 16), rd<uint64_t>(m + 24)}; return true;
      case 'M': e = {Ev::Modify, 0, rd<uint32_t>(m + 24), rd<uint32_t>(m + 28), rd<uint64_t>(m + 16), 0}; return true;
      default: return false;   // 'R', 'P': stored, no book effect
    }
  }
};

// Same wire format, resolved the PITCH way: the LH ignores the locate and keeps an
// order-id -> symbol map fed by the Add messages. Exercises LineHandler3's second path.
struct ToyL3NoLocate : ToyL3 {
  static constexpr uint8_t kProtoId   = 3;
  static constexpr bool    kHasLocate = false;
};

}  // namespace rawmd::l3
