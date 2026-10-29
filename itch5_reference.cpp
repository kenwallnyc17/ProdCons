//#pragma once
// itch5_reference.cpp — a deliberately simple, spec-conformant NASDAQ TotalView-ITCH 5.0
// order-book builder, meant as a golden reference to diff against an optimized implementation.
//
// Priorities, in order: wire-format correctness, deterministic output, legibility. Not speed.
// Containers are std::map / std::unordered_map on purpose.
//
// Build:
//   g++ -std=c++23 -O2 -Wall -Wextra -o itch5_reference itch5_reference.cpp
//
// Self-test (golden wire-format byte vectors, independent of the decoders' own structs):
//   ./itch5_reference --self-test
//
// Typical use:
//   ./itch5_reference --in 01302019.NASDAQ_ITCH50 --events ref.csv --snapshot ref.snap.csv
//   ./itch5_reference --in capture.pcap --events ref.csv --levels 5 --symbol MSFT
//   diff <(sort ref.csv) <(sort fast.csv)
//
// Input formats (--format auto picks by pcap magic, else BinaryFILE):
//   binaryfile  NASDAQ's downloadable session files: repeated [2-byte BE length][message].
//   pcap        classic libpcap (us or ns), Ethernet/VLAN or raw IPv4, UDP, MoldUDP64 framed.
//
// Output contract — everything here exists to make `diff` meaningful:
//   * Prices are RAW ITCH integers (4 implied decimals). No floating point anywhere in the
//     output path, so formatting can never account for a discrepancy.
//   * An absent price level prints as "0,0". Price 0 is not a valid quote.
//   * One event line per book-changing message that resolved to a symbol ('A','F','E','C',
//     'X','D','U'). Non-book messages never emit.
//   * "seq" is the MoldUDP64 sequence number of that message for pcap input, or the 1-based
//     message ordinal for BinaryFILE. Either way it is a stable join key between two runs.
//   * --publish on-change (default) emits only when the published view actually changes;
//     --publish always emits one line per book-changing message.
//   * The snapshot iterates symbols in stock_locate order and levels in price order.
//   * Anomaly counters are part of the comparison. If two implementations agree on the books
//     but disagree on, say, unknown_order_ref, one of them is quietly dropping messages.
//
// Round-lot best bid/offer has two defensible definitions from order-by-order data and they
// diverge exactly in odd-lot-heavy names. Pick the one your fast path implements:
//   --rl-policy aggregate  (default) a level qualifies when the SUM of resting shares at that
//                          price is >= round_lot_size; published size is the full level.
//   --rl-policy order      a level qualifies when at least one resting order is individually
//                          >= round_lot_size; published size counts only such orders.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

// ============================================================================
// Big-endian field readers.
//
// Assembled byte by byte from an unaligned pointer. No packed structs, no
// reinterpret_cast, no host-endianness assumption, no strict-aliasing or
// alignment UB. GCC folds each of these into a single movbe/bswap at -O2.
// ============================================================================

constexpr uint8_t be8(const uint8_t* p) { return p[0]; }

constexpr uint16_t be16(const uint8_t* p) {
  return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) << 8 | p[1]);
}

constexpr uint32_t be32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) << 24 | static_cast<uint32_t>(p[1]) << 16 |
         static_cast<uint32_t>(p[2]) << 8 | static_cast<uint32_t>(p[3]);
}

// ITCH timestamps are 6 bytes: nanoseconds since midnight. This is the field whose
// width is most often gotten wrong; an 8-byte timestamp shifts every later field.
constexpr uint64_t be48(const uint8_t* p) {
  return static_cast<uint64_t>(p[0]) << 40 | static_cast<uint64_t>(p[1]) << 32 |
         static_cast<uint64_t>(p[2]) << 24 | static_cast<uint64_t>(p[3]) << 16 |
         static_cast<uint64_t>(p[4]) << 8 | static_cast<uint64_t>(p[5]);
}

constexpr uint64_t be64(const uint8_t* p) {
  return static_cast<uint64_t>(be32(p)) << 32 | be32(p + 4);
}

// ============================================================================
// ITCH 5.0 message sizes, including the one-byte type.
// ============================================================================

constexpr std::array<uint8_t, 256> kMessageSize = [] {
  std::array<uint8_t, 256> t{};
  t[static_cast<unsigned char>('S')] = 12;  // System Event
  t[static_cast<unsigned char>('R')] = 39;  // Stock Directory
  t[static_cast<unsigned char>('H')] = 25;  // Stock Trading Action
  t[static_cast<unsigned char>('Y')] = 20;  // Reg SHO Restriction
  t[static_cast<unsigned char>('L')] = 26;  // Market Participant Position
  t[static_cast<unsigned char>('V')] = 35;  // MWCB Decline Level
  t[static_cast<unsigned char>('W')] = 12;  // MWCB Status
  t[static_cast<unsigned char>('K')] = 28;  // IPO Quoting Period Update
  t[static_cast<unsigned char>('J')] = 35;  // LULD Auction Collar
  t[static_cast<unsigned char>('h')] = 21;  // Operational Halt
  t[static_cast<unsigned char>('A')] = 36;  // Add Order, no attribution
  t[static_cast<unsigned char>('F')] = 40;  // Add Order with MPID
  t[static_cast<unsigned char>('E')] = 31;  // Order Executed
  t[static_cast<unsigned char>('C')] = 36;  // Order Executed with Price
  t[static_cast<unsigned char>('X')] = 23;  // Order Cancel
  t[static_cast<unsigned char>('D')] = 19;  // Order Delete
  t[static_cast<unsigned char>('U')] = 35;  // Order Replace
  t[static_cast<unsigned char>('P')] = 44;  // Trade, non-cross
  t[static_cast<unsigned char>('Q')] = 40;  // Cross Trade
  t[static_cast<unsigned char>('B')] = 19;  // Broken Trade
  t[static_cast<unsigned char>('I')] = 50;  // NOII
  t[static_cast<unsigned char>('N')] = 20;  // RPII
  return t;
}();

static_assert(kMessageSize[static_cast<unsigned char>('A')] == 36);
static_assert(kMessageSize[static_cast<unsigned char>('F')] == 40);
static_assert(kMessageSize[static_cast<unsigned char>('E')] == 31);
static_assert(kMessageSize[static_cast<unsigned char>('C')] == 36);
static_assert(kMessageSize[static_cast<unsigned char>('X')] == 23);
static_assert(kMessageSize[static_cast<unsigned char>('D')] == 19);
static_assert(kMessageSize[static_cast<unsigned char>('U')] == 35);
static_assert(kMessageSize[static_cast<unsigned char>('R')] == 39);

constexpr size_t message_size(uint8_t type) { return kMessageSize[type]; }

// Field offsets, straight from the spec tables. Shared prefix for every message:
//   0 type(1)  1 stock_locate(2)  3 tracking_number(2)  5 timestamp(6)
namespace off {
constexpr size_t kLocate = 1;
constexpr size_t kTimestamp = 5;

// 'A' / 'F'
constexpr size_t kAddRef = 11, kAddSide = 19, kAddShares = 20, kAddStock = 24, kAddPrice = 32;
// 'E' / 'C'
constexpr size_t kExecRef = 11, kExecShares = 19;
// 'X'
constexpr size_t kCancelRef = 11, kCancelShares = 19;
// 'D'
constexpr size_t kDeleteRef = 11;
// 'U'
constexpr size_t kReplaceOldRef = 11, kReplaceNewRef = 19, kReplaceShares = 27,
                 kReplacePrice = 31;
// 'R'
constexpr size_t kDirStock = 11, kDirRoundLot = 21;
// 'H'
constexpr size_t kActionStock = 11, kActionState = 19;
// 'S'
constexpr size_t kSysEvent = 11;
}  // namespace off

// ============================================================================
// Book state
// ============================================================================

using Price = uint32_t;  // raw ITCH price, 4 implied decimals

struct Level {
  uint64_t total_shares = 0;  // all resting shares at this price
  uint64_t rl_shares = 0;     // shares from orders individually >= round_lot_size
  uint32_t orders = 0;
};

// One container type for both sides; bids are read in reverse. Keeping a single
// comparator avoids duplicating every book routine.
using Book = std::map<Price, Level>;

struct Order {
  Price price = 0;
  uint32_t shares = 0;
  uint16_t locate = 0;
  char side = 0;  // 'B' or 'S'
};

struct SymbolState {
  std::array<char, 9> symbol{};  // NUL-terminated, trailing spaces trimmed
  uint32_t round_lot_size = 100;
  bool directory_seen = false;
  char trading_state = '-';  // '-' until an 'H' Stock Trading Action arrives
  Book bids;
  Book asks;
  std::string last_view;  // for --publish on-change
};

enum class RoundLotPolicy { Aggregate, Order };

struct Stats {
  uint64_t pcap_packets = 0;
  uint64_t udp_datagrams = 0;
  uint64_t non_udp_packets = 0;
  uint64_t datagram_framing_errors = 0;
  uint64_t heartbeats = 0;
  uint64_t end_of_session = 0;

  uint64_t messages = 0;
  std::array<uint64_t, 256> by_type{};
  uint64_t unknown_type = 0;
  uint64_t truncated_message = 0;

  uint64_t seq_gaps = 0;           // number of discontinuities
  uint64_t seq_gap_messages = 0;   // messages presumed missed
  uint64_t seq_regressions = 0;    // retransmits / duplicates

  uint64_t duplicate_order_ref = 0;
  uint64_t unknown_order_ref = 0;
  uint64_t exec_overfill = 0;      // executed more than the order held
  uint64_t cancel_overfill = 0;
  uint64_t zero_quantity = 0;
  uint64_t missing_level = 0;      // removal from a price with no level: book was wrong
  uint64_t level_underflow = 0;    // level held fewer shares than the removal
  uint64_t late_directory = 0;     // 'R' after orders rested: stale rl_shares possible
  uint64_t bad_side = 0;
  uint64_t crossed_books = 0;      // best bid >= best ask after an update

  uint64_t events_emitted = 0;
};

// ============================================================================
// Engine
// ============================================================================

class BookBuilder {
 public:
  struct Options {
    size_t levels = 5;
    RoundLotPolicy rl_policy = RoundLotPolicy::Aggregate;
    bool publish_on_change = true;
    std::set<std::string> symbol_filter;  // empty = all
    uint64_t max_messages = 0;            // 0 = unlimited
  };

  BookBuilder(Options opt, std::ostream* events) : opt_(std::move(opt)), events_(events) {
    if (events_) write_event_header();
  }

  const Stats& stats() const { return stats_; }
  bool message_budget_exhausted() const {
    return opt_.max_messages != 0 && stats_.messages >= opt_.max_messages;
  }

  // ---- framing -------------------------------------------------------------

  // A MoldUDP64 datagram: session[10] sequence[8] message_count[2], then
  // repeated [length(2)][message]. Returns messages consumed.
  size_t on_mold_datagram(const uint8_t* p, size_t len) {
    ++stats_.udp_datagrams;
    if (len < 20) {
      ++stats_.datagram_framing_errors;
      return 0;
    }
    const uint64_t seq = be64(p + 10);
    const uint16_t count = be16(p + 18);

    if (count == 0xFFFF) {  // end-of-session marker, carries no messages
      ++stats_.end_of_session;
      return 0;
    }
    if (count == 0) {
      ++stats_.heartbeats;
      check_sequence(seq, 0);
      return 0;
    }

    // Validate the whole datagram before mutating any state, so a malformed
    // tail cannot leave the book half-updated.
    size_t offset = 20;
    for (uint16_t i = 0; i < count; ++i) {
      if (len - offset < 2) {
        ++stats_.datagram_framing_errors;
        return 0;
      }
      const size_t msg_len = be16(p + offset);
      offset += 2;
      if (msg_len == 0 || msg_len > len - offset) {
        ++stats_.datagram_framing_errors;
        return 0;
      }
      offset += msg_len;
    }
    if (offset != len) {  // trailing bytes: not a MoldUDP64 datagram we understand
      ++stats_.datagram_framing_errors;
      return 0;
    }

    check_sequence(seq, count);

    offset = 20;
    size_t consumed = 0;
    for (uint16_t i = 0; i < count; ++i) {
      const size_t msg_len = be16(p + offset);
      offset += 2;
      on_message(p + offset, msg_len, seq + i);
      offset += msg_len;
      ++consumed;
      if (message_budget_exhausted()) break;
    }
    return consumed;
  }

  // ---- one ITCH message ----------------------------------------------------

  void on_message(const uint8_t* p, size_t len, uint64_t seq) {
    if (len == 0) return;
    ++stats_.messages;
    const uint8_t type = p[0];
    ++stats_.by_type[type];

    const size_t expect = message_size(type);
    if (expect == 0) {
      ++stats_.unknown_type;
      return;
    }
    if (len < expect) {
      ++stats_.truncated_message;
      return;
    }

    switch (static_cast<char>(type)) {
      case 'R': on_stock_directory(p); return;
      case 'H': on_trading_action(p); return;
      case 'S': system_event_ = static_cast<char>(be8(p + off::kSysEvent)); return;

      case 'A':
      case 'F': on_add_order(p, seq); return;
      case 'E':
      case 'C': on_order_executed(p, seq); return;
      case 'X': on_order_cancel(p, seq); return;
      case 'D': on_order_delete(p, seq); return;
      case 'U': on_order_replace(p, seq); return;

      default: return;  // counted, no book effect
    }
  }

  // ---- snapshot ------------------------------------------------------------

  void write_snapshot(std::ostream& out) const {
    out << "# itch5_reference snapshot\n";
    out << "# levels=" << opt_.levels << " rl_policy="
        << (opt_.rl_policy == RoundLotPolicy::Aggregate ? "aggregate" : "order") << "\n";
    out << "# prices are raw ITCH integers (4 implied decimals); 0,0 = absent level\n";
    out << snapshot_header() << "\n";
    for (const auto& [locate, state] : symbols_) {  // std::map: locate order
      if (!state.directory_seen && state.bids.empty() && state.asks.empty()) continue;
      if (!symbol_selected(state)) continue;
      std::string line;
      append_u64(line, locate);
      line += ',';
      line += state.symbol.data();
      line += ',';
      append_u64(line, state.round_lot_size);
      line += ',';
      line += state.trading_state;
      line += ',';
      append_u64(line, count_orders(state.bids));
      line += ',';
      append_u64(line, count_orders(state.asks));
      line += ',';
      append_view(line, state);
      out << line << '\n';
    }
  }

  void write_stats(std::ostream& out) const {
    const Stats& s = stats_;
    out << "# itch5_reference counters (these are part of the comparison)\n";
    auto row = [&out](const char* k, uint64_t v) { out << k << '\t' << v << '\n'; };
    row("pcap_packets", s.pcap_packets);
    row("non_udp_packets", s.non_udp_packets);
    row("udp_datagrams", s.udp_datagrams);
    row("datagram_framing_errors", s.datagram_framing_errors);
    row("heartbeats", s.heartbeats);
    row("end_of_session", s.end_of_session);
    row("messages", s.messages);
    row("unknown_type", s.unknown_type);
    row("truncated_message", s.truncated_message);
    row("seq_gaps", s.seq_gaps);
    row("seq_gap_messages", s.seq_gap_messages);
    row("seq_regressions", s.seq_regressions);
    row("duplicate_order_ref", s.duplicate_order_ref);
    row("unknown_order_ref", s.unknown_order_ref);
    row("exec_overfill", s.exec_overfill);
    row("cancel_overfill", s.cancel_overfill);
    row("zero_quantity", s.zero_quantity);
    row("missing_level", s.missing_level);
    row("level_underflow", s.level_underflow);
    row("late_directory", s.late_directory);
    row("bad_side", s.bad_side);
    row("crossed_books", s.crossed_books);
    row("live_orders", static_cast<uint64_t>(orders_.size()));
    row("symbols_with_state", count_active_symbols());
    row("events_emitted", s.events_emitted);
    out << "# messages by type (printable types only)\n";
    for (size_t i = 0; i < 256; ++i) {
      if (s.by_type[i] == 0) continue;
      out << "type_" << static_cast<char>(i) << '\t' << s.by_type[i] << '\n';
    }
  }

 private:
  // ---- sequence tracking ---------------------------------------------------

  void check_sequence(uint64_t seq, uint16_t count) {
    if (!expected_seq_) {
      expected_seq_ = seq;
    } else if (seq > *expected_seq_) {
      ++stats_.seq_gaps;
      stats_.seq_gap_messages += seq - *expected_seq_;
    } else if (seq < *expected_seq_) {
      ++stats_.seq_regressions;
    }
    const uint64_t next = seq + count;
    if (!expected_seq_ || next > *expected_seq_) expected_seq_ = next;
  }

  // ---- level maintenance ---------------------------------------------------

  static bool rl_eligible(uint32_t shares, uint32_t round_lot) {
    return round_lot != 0 && shares >= round_lot;
  }

  void level_add(Book& book, Price price, uint32_t shares, uint32_t round_lot) {
    Level& lvl = book[price];
    lvl.total_shares += shares;
    if (rl_eligible(shares, round_lot)) lvl.rl_shares += shares;
    ++lvl.orders;
  }

  // Remove an order entirely.
  void level_remove(Book& book, Price price, uint32_t shares, uint32_t round_lot) {
    auto it = book.find(price);
    if (it == book.end()) {
      ++stats_.missing_level;
      return;
    }
    Level& lvl = it->second;
    if (lvl.total_shares < shares) {
      ++stats_.level_underflow;
      lvl.total_shares = 0;
    } else {
      lvl.total_shares -= shares;
    }
    if (rl_eligible(shares, round_lot)) {
      lvl.rl_shares = (lvl.rl_shares < shares) ? 0 : lvl.rl_shares - shares;
    }
    if (lvl.orders > 0) --lvl.orders;
    if (lvl.orders == 0 || lvl.total_shares == 0) book.erase(it);
  }

  // Resize an order in place (partial execution or partial cancel). The order keeps
  // its slot at the level, but round-lot eligibility is re-evaluated at the new size.
  void level_resize(Book& book, Price price, uint32_t old_shares, uint32_t new_shares,
                    uint32_t round_lot) {
    if (new_shares == 0) {
      level_remove(book, price, old_shares, round_lot);
      return;
    }
    auto it = book.find(price);
    if (it == book.end()) {
      ++stats_.missing_level;
      return;
    }
    Level& lvl = it->second;
    const uint32_t delta = old_shares - new_shares;  // caller guarantees new <= old
    if (lvl.total_shares < delta) {
      ++stats_.level_underflow;
      lvl.total_shares = 0;
    } else {
      lvl.total_shares -= delta;
    }
    const uint64_t was = rl_eligible(old_shares, round_lot) ? old_shares : 0;
    const uint64_t now = rl_eligible(new_shares, round_lot) ? new_shares : 0;
    if (was > now) {
      const uint64_t drop = was - now;
      lvl.rl_shares = (lvl.rl_shares < drop) ? 0 : lvl.rl_shares - drop;
    } else {
      lvl.rl_shares += now - was;
    }
    if (lvl.total_shares == 0 || lvl.orders == 0) book.erase(it);
  }

  Book& book_for(SymbolState& state, char side) {
    return side == 'B' ? state.bids : state.asks;
  }

  // ---- message handlers ----------------------------------------------------

  SymbolState& state_for(uint16_t locate) { return symbols_[locate]; }

  void on_stock_directory(const uint8_t* p) {
    const uint16_t locate = be16(p + off::kLocate);
    SymbolState& state = state_for(locate);
    if (!state.bids.empty() || !state.asks.empty()) ++stats_.late_directory;
    set_symbol(state, p + off::kDirStock);
    state.round_lot_size = be32(p + off::kDirRoundLot);
    state.directory_seen = true;
  }

  void on_trading_action(const uint8_t* p) {
    const uint16_t locate = be16(p + off::kLocate);
    SymbolState& state = state_for(locate);
    if (state.symbol[0] == '\0') set_symbol(state, p + off::kActionStock);
    state.trading_state = static_cast<char>(be8(p + off::kActionState));
  }

  void on_add_order(const uint8_t* p, uint64_t seq) {
    const uint16_t locate = be16(p + off::kLocate);
    const uint64_t ref = be64(p + off::kAddRef);
    const char side = static_cast<char>(be8(p + off::kAddSide));
    const uint32_t shares = be32(p + off::kAddShares);
    const Price price = be32(p + off::kAddPrice);

    if (side != 'B' && side != 'S') {
      ++stats_.bad_side;
      return;
    }
    if (shares == 0) ++stats_.zero_quantity;

    SymbolState& state = state_for(locate);
    if (state.symbol[0] == '\0') set_symbol(state, p + off::kAddStock);

    auto [it, inserted] = orders_.try_emplace(ref, Order{price, shares, locate, side});
    if (!inserted) {
      // A duplicate reference means our view and the feed have already diverged.
      // Retire the stale order so its shares do not leak into the book forever.
      ++stats_.duplicate_order_ref;
      const Order& stale = it->second;
      SymbolState& stale_state = state_for(stale.locate);
      level_remove(book_for(stale_state, stale.side), stale.price, stale.shares,
                   stale_state.round_lot_size);
      it->second = Order{price, shares, locate, side};
    }

    level_add(book_for(state, side), price, shares, state.round_lot_size);
    publish(locate, state, static_cast<char>(p[0]), seq);
  }

  // 'E' and 'C' have identical book effects. 'C' also carries a printable flag and an
  // execution price, but those describe the print, not the resting order.
  void on_order_executed(const uint8_t* p, uint64_t seq) {
    const uint64_t ref = be64(p + off::kExecRef);
    const uint32_t exec = be32(p + off::kExecShares);
    reduce_order(ref, exec, static_cast<char>(p[0]), seq, stats_.exec_overfill);
  }

  void on_order_cancel(const uint8_t* p, uint64_t seq) {
    const uint64_t ref = be64(p + off::kCancelRef);
    const uint32_t cancelled = be32(p + off::kCancelShares);
    reduce_order(ref, cancelled, static_cast<char>(p[0]), seq, stats_.cancel_overfill);
  }

  void reduce_order(uint64_t ref, uint32_t qty, char msg_type, uint64_t seq,
                    uint64_t& overfill_counter) {
    auto it = orders_.find(ref);
    if (it == orders_.end()) {
      ++stats_.unknown_order_ref;
      return;
    }
    if (qty == 0) ++stats_.zero_quantity;

    Order& ord = it->second;
    SymbolState& state = state_for(ord.locate);

    // Clamp instead of letting unsigned arithmetic wrap. An overfill means we
    // missed the add or an earlier reduction, so the order is already suspect.
    uint32_t applied = qty;
    if (applied > ord.shares) {
      ++overfill_counter;
      applied = ord.shares;
    }
    const uint32_t remaining = ord.shares - applied;

    level_resize(book_for(state, ord.side), ord.price, ord.shares, remaining,
                 state.round_lot_size);

    const uint16_t locate = ord.locate;
    if (remaining == 0) {
      orders_.erase(it);
    } else {
      ord.shares = remaining;
    }
    publish(locate, state_for(locate), msg_type, seq);
  }

  void on_order_delete(const uint8_t* p, uint64_t seq) {
    const uint64_t ref = be64(p + off::kDeleteRef);
    auto it = orders_.find(ref);
    if (it == orders_.end()) {
      ++stats_.unknown_order_ref;
      return;
    }
    const Order ord = it->second;
    orders_.erase(it);
    SymbolState& state = state_for(ord.locate);
    level_remove(book_for(state, ord.side), ord.price, ord.shares, state.round_lot_size);
    publish(ord.locate, state, static_cast<char>(p[0]), seq);
  }

  // Replace retires the original reference and creates a new one. Side and symbol are
  // inherited from the original order; only size and price come from the message.
  void on_order_replace(const uint8_t* p, uint64_t seq) {
    const uint64_t old_ref = be64(p + off::kReplaceOldRef);
    const uint64_t new_ref = be64(p + off::kReplaceNewRef);
    const uint32_t shares = be32(p + off::kReplaceShares);
    const Price price = be32(p + off::kReplacePrice);

    auto it = orders_.find(old_ref);
    if (it == orders_.end()) {
      ++stats_.unknown_order_ref;
      return;
    }
    const Order old_ord = it->second;
    orders_.erase(it);

    SymbolState& state = state_for(old_ord.locate);
    level_remove(book_for(state, old_ord.side), old_ord.price, old_ord.shares,
                 state.round_lot_size);

    if (shares == 0) ++stats_.zero_quantity;
    auto [nit, inserted] =
        orders_.try_emplace(new_ref, Order{price, shares, old_ord.locate, old_ord.side});
    if (!inserted) {
      ++stats_.duplicate_order_ref;
      const Order& stale = nit->second;
      SymbolState& stale_state = state_for(stale.locate);
      level_remove(book_for(stale_state, stale.side), stale.price, stale.shares,
                   stale_state.round_lot_size);
      nit->second = Order{price, shares, old_ord.locate, old_ord.side};
    }
    level_add(book_for(state, old_ord.side), price, shares, state.round_lot_size);
    publish(old_ord.locate, state, static_cast<char>(p[0]), seq);
  }

  // ---- symbol bookkeeping --------------------------------------------------

  static void set_symbol(SymbolState& state, const uint8_t* stock8) {
    size_t n = 8;
    while (n > 0 && stock8[n - 1] == ' ') --n;  // ITCH pads with spaces
    std::memcpy(state.symbol.data(), stock8, n);
    state.symbol[n] = '\0';
  }

  bool symbol_selected(const SymbolState& state) const {
    if (opt_.symbol_filter.empty()) return true;
    return opt_.symbol_filter.contains(state.symbol.data());
  }

  static uint64_t count_orders(const Book& book) {
    uint64_t n = 0;
    for (const auto& [price, lvl] : book) n += lvl.orders;
    return n;
  }

  uint64_t count_active_symbols() const {
    uint64_t n = 0;
    for (const auto& [locate, state] : symbols_)
      if (state.directory_seen || !state.bids.empty() || !state.asks.empty()) ++n;
    return n;
  }

  // ---- output --------------------------------------------------------------

  static void append_u64(std::string& out, uint64_t v) {
    char buf[20];
    char* end = buf + sizeof(buf);
    char* cur = end;
    do {
      *--cur = static_cast<char>('0' + v % 10);
      v /= 10;
    } while (v != 0);
    out.append(cur, static_cast<size_t>(end - cur));
  }

  std::string snapshot_header() const {
    std::string h = "locate,symbol,round_lot,trading_state,bid_orders,ask_orders";
    append_view_header(h);
    return h;
  }

  void append_view_header(std::string& h) const {
    for (size_t i = 1; i <= opt_.levels; ++i) {
      h += ",bid_px_" + std::to_string(i) + ",bid_sz_" + std::to_string(i);
    }
    for (size_t i = 1; i <= opt_.levels; ++i) {
      h += ",ask_px_" + std::to_string(i) + ",ask_sz_" + std::to_string(i);
    }
    h += ",rl_bid_px,rl_bid_sz,rl_ask_px,rl_ask_sz";
  }

  void write_event_header() {
    *events_ << "# itch5_reference events\n";
    *events_ << "# levels=" << opt_.levels << " rl_policy="
             << (opt_.rl_policy == RoundLotPolicy::Aggregate ? "aggregate" : "order")
             << " publish=" << (opt_.publish_on_change ? "on-change" : "always") << "\n";
    *events_ << "# prices are raw ITCH integers (4 implied decimals); 0,0 = absent level\n";
    std::string h = "seq,type,locate,symbol";
    append_view_header(h);
    *events_ << h << "\n";
  }

  // The published view: top-N both sides plus the round-lot best. Bids descend,
  // asks ascend; both come straight from std::map order, so two runs cannot differ
  // in ordering.
  void append_view(std::string& out, const SymbolState& state) const {
    size_t emitted = 0;
    for (auto it = state.bids.rbegin(); it != state.bids.rend() && emitted < opt_.levels;
         ++it, ++emitted) {
      append_u64(out, it->first);
      out += ',';
      append_u64(out, it->second.total_shares);
      out += ',';
    }
    for (; emitted < opt_.levels; ++emitted) out += "0,0,";

    emitted = 0;
    for (auto it = state.asks.begin(); it != state.asks.end() && emitted < opt_.levels;
         ++it, ++emitted) {
      append_u64(out, it->first);
      out += ',';
      append_u64(out, it->second.total_shares);
      out += ',';
    }
    for (; emitted < opt_.levels; ++emitted) out += "0,0,";

    const auto [bid_px, bid_sz] = round_lot_best(state, 'B');
    const auto [ask_px, ask_sz] = round_lot_best(state, 'S');
    append_u64(out, bid_px);
    out += ',';
    append_u64(out, bid_sz);
    out += ',';
    append_u64(out, ask_px);
    out += ',';
    append_u64(out, ask_sz);
  }

  std::pair<Price, uint64_t> round_lot_best(const SymbolState& state, char side) const {
    const uint32_t rl = state.round_lot_size;
    auto qualifies = [&](const Level& lvl) -> uint64_t {
      if (opt_.rl_policy == RoundLotPolicy::Aggregate) {
        return (rl != 0 && lvl.total_shares >= rl) ? lvl.total_shares : 0;
      }
      return lvl.rl_shares;  // only individually round-lot-sized orders count
    };
    if (side == 'B') {
      for (auto it = state.bids.rbegin(); it != state.bids.rend(); ++it) {
        if (const uint64_t sz = qualifies(it->second); sz > 0) return {it->first, sz};
      }
    } else {
      for (auto it = state.asks.begin(); it != state.asks.end(); ++it) {
        if (const uint64_t sz = qualifies(it->second); sz > 0) return {it->first, sz};
      }
    }
    return {0, 0};
  }

  void publish(uint16_t locate, SymbolState& state, char msg_type, uint64_t seq) {
    if (!state.bids.empty() && !state.asks.empty() &&
        state.bids.rbegin()->first >= state.asks.begin()->first) {
      ++stats_.crossed_books;
    }
    if (!events_ || !symbol_selected(state)) return;

    std::string view;
    append_view(view, state);
    if (opt_.publish_on_change) {
      if (view == state.last_view) return;
      state.last_view = view;
    }

    std::string line;
    append_u64(line, seq);
    line += ',';
    line += msg_type;
    line += ',';
    append_u64(line, locate);
    line += ',';
    line += state.symbol.data();
    line += ',';
    line += view;
    line += '\n';
    events_->write(line.data(), static_cast<std::streamsize>(line.size()));
    ++stats_.events_emitted;
  }

  Options opt_;
  std::ostream* events_;
  std::map<uint16_t, SymbolState> symbols_;       // locate order, lazily populated
  std::unordered_map<uint64_t, Order> orders_;    // live order references
  std::optional<uint64_t> expected_seq_;
  char system_event_ = ' ';
  Stats stats_{};
};

// ============================================================================
// Input: NASDAQ BinaryFILE  — repeated [2-byte BE length][message]
// ============================================================================

void read_binaryfile(std::istream& in, BookBuilder& engine) {
  std::vector<uint8_t> buf(1024);
  uint64_t ordinal = 1;
  uint8_t len_bytes[2];
  while (true) {
    in.read(reinterpret_cast<char*>(len_bytes), 2);
    const std::streamsize got = in.gcount();
    if (got == 0) break;  // clean end of file
    if (got != 2) throw std::runtime_error("BinaryFILE: truncated length prefix");

    const size_t msg_len = be16(len_bytes);
    if (msg_len == 0) throw std::runtime_error("BinaryFILE: zero-length message");
    if (msg_len > buf.size()) buf.resize(msg_len);
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(msg_len));
    if (static_cast<size_t>(in.gcount()) != msg_len)
      throw std::runtime_error("BinaryFILE: truncated message body");

    engine.on_message(buf.data(), msg_len, ordinal);
    ++ordinal;
    if (engine.message_budget_exhausted()) break;
  }
}

// ============================================================================
// Input: classic pcap -> UDP payload -> MoldUDP64
// ============================================================================

constexpr uint32_t kMaxRecord = 64u * 1024 * 1024;

uint16_t le_or_be16(const uint8_t* p, bool little) {
  return little ? static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8)) : be16(p);
}

uint32_t le_or_be32(const uint8_t* p, bool little) {
  if (!little) return be32(p);
  return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 |
         static_cast<uint32_t>(p[2]) << 16 | static_cast<uint32_t>(p[3]) << 24;
}

// Locates the UDP payload. Returns false for anything that is not a complete,
// unfragmented IPv4 UDP datagram, which the caller counts rather than treats as an error.
bool udp_payload(const uint8_t* pkt, size_t pkt_len, uint32_t link_type, size_t& out_off,
                 size_t& out_len) {
  size_t net = 0;
  if (link_type == 1) {  // Ethernet
    if (pkt_len < 14) return false;
    uint16_t ether = be16(pkt + 12);
    net = 14;
    while (ether == 0x8100 || ether == 0x88A8 || ether == 0x9100) {  // VLAN / QinQ
      if (pkt_len - net < 4) return false;
      ether = be16(pkt + net + 2);
      net += 4;
    }
    if (ether != 0x0800) return false;  // not IPv4
  } else if (link_type == 101 || link_type == 12) {  // raw IP
    if (pkt_len == 0) return false;
  } else {
    return false;
  }

  if (pkt_len < net + 20) return false;
  const uint8_t* ip = pkt + net;
  if ((ip[0] >> 4) != 4) return false;
  const size_t ihl = static_cast<size_t>(ip[0] & 0x0F) * 4;
  if (ihl < 20 || pkt_len - net < ihl) return false;
  const size_t total = be16(ip + 2);
  if (total < ihl + 8 || total > pkt_len - net) return false;
  if (ip[9] != 17) return false;                      // not UDP
  if ((be16(ip + 6) & 0x3FFF) != 0) return false;     // fragmented

  const size_t udp_off = net + ihl;
  const uint8_t* udp = pkt + udp_off;
  const size_t udp_len = be16(udp + 4);
  if (udp_len < 8 || udp_len > total - ihl) return false;

  out_off = udp_off + 8;
  out_len = udp_len - 8;
  return true;
}

void read_pcap(std::istream& in, BookBuilder& engine, Stats& stats) {
  uint8_t gh[24];
  in.read(reinterpret_cast<char*>(gh), sizeof(gh));
  if (static_cast<size_t>(in.gcount()) != sizeof(gh))
    throw std::runtime_error("pcap: incomplete global header");

  bool little;
  switch (be32(gh)) {
    case 0xD4C3B2A1:
    case 0x4D3CB2A1: little = true; break;
    case 0xA1B2C3D4:
    case 0xA1B23C4D: little = false; break;
    default: throw std::runtime_error("pcap: bad magic number");
  }
  if (le_or_be16(gh + 4, little) != 2 || le_or_be16(gh + 6, little) != 4)
    throw std::runtime_error("pcap: expected version 2.4");

  const uint32_t snaplen = le_or_be32(gh + 16, little);
  const uint32_t link_type = le_or_be32(gh + 20, little) & 0xFFFF;
  if (snaplen == 0 || snaplen > kMaxRecord)
    throw std::runtime_error("pcap: implausible snaplen");
  if (link_type != 1 && link_type != 12 && link_type != 101)
    throw std::runtime_error("pcap: unsupported link type " + std::to_string(link_type));

  std::vector<uint8_t> pkt(snaplen);  // one buffer, reused
  uint8_t rh[16];
  while (true) {
    in.read(reinterpret_cast<char*>(rh), sizeof(rh));
    const std::streamsize got = in.gcount();
    if (got == 0) break;
    if (got != static_cast<std::streamsize>(sizeof(rh)))
      throw std::runtime_error("pcap: incomplete record header");

    const uint32_t cap_len = le_or_be32(rh + 8, little);
    const uint32_t orig_len = le_or_be32(rh + 12, little);
    if (cap_len > orig_len || cap_len > snaplen || cap_len > kMaxRecord)
      throw std::runtime_error("pcap: record exceeds snaplen");

    if (cap_len > pkt.size()) pkt.resize(cap_len);
    in.read(reinterpret_cast<char*>(pkt.data()), static_cast<std::streamsize>(cap_len));
    if (static_cast<size_t>(in.gcount()) != cap_len)
      throw std::runtime_error("pcap: truncated record body");
    ++stats.pcap_packets;

    size_t off = 0, len = 0;
    if (udp_payload(pkt.data(), cap_len, link_type, off, len)) {
      engine.on_mold_datagram(pkt.data() + off, len);
    } else {
      ++stats.non_udp_packets;
    }
    if (engine.message_budget_exhausted()) break;
  }
}

// ============================================================================
// Self-test: golden wire-format bytes at spec offsets.
//
// These vectors are written with independent little helpers, not with the decoder's
// own notion of layout, so a layout error cannot hide by being symmetric. This is
// exactly the failure mode of a suite that builds its inputs from the same structs
// it decodes with.
// ============================================================================

void put16(uint8_t* p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v >> 8);
  p[1] = static_cast<uint8_t>(v);
}
void put32(uint8_t* p, uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = static_cast<uint8_t>(v >> (24 - 8 * i));
}
void put48(uint8_t* p, uint64_t v) {
  for (int i = 0; i < 6; ++i) p[i] = static_cast<uint8_t>(v >> (40 - 8 * i));
}
void put64(uint8_t* p, uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (56 - 8 * i));
}

std::vector<uint8_t> make_directory(uint16_t locate, const char* sym, uint32_t round_lot) {
  std::vector<uint8_t> m(39, 0);
  m[0] = 'R';
  put16(&m[1], locate);
  put16(&m[3], 1);
  put48(&m[5], 1'000'000);
  std::memset(&m[11], ' ', 8);
  std::memcpy(&m[11], sym, std::min<size_t>(8, std::strlen(sym)));
  m[19] = 'Q';
  m[20] = 'N';
  put32(&m[21], round_lot);
  m[25] = 'N';
  return m;
}

std::vector<uint8_t> make_add(uint16_t locate, uint64_t ref, char side, uint32_t shares,
                              const char* sym, uint32_t price, bool with_mpid = false) {
  std::vector<uint8_t> m(with_mpid ? 40 : 36, 0);
  m[0] = with_mpid ? 'F' : 'A';
  put16(&m[1], locate);
  put16(&m[3], 1);
  put48(&m[5], 2'000'000);
  put64(&m[11], ref);
  m[19] = static_cast<uint8_t>(side);
  put32(&m[20], shares);
  std::memset(&m[24], ' ', 8);
  std::memcpy(&m[24], sym, std::min<size_t>(8, std::strlen(sym)));
  put32(&m[32], price);
  if (with_mpid) std::memcpy(&m[36], "NSDQ", 4);
  return m;
}

std::vector<uint8_t> make_exec(uint64_t ref, uint32_t shares) {
  std::vector<uint8_t> m(31, 0);
  m[0] = 'E';
  put16(&m[1], 1);
  put16(&m[3], 1);
  put48(&m[5], 3'000'000);
  put64(&m[11], ref);
  put32(&m[19], shares);
  put64(&m[23], 99);
  return m;
}

std::vector<uint8_t> make_cancel(uint64_t ref, uint32_t shares) {
  std::vector<uint8_t> m(23, 0);
  m[0] = 'X';
  put16(&m[1], 1);
  put16(&m[3], 1);
  put48(&m[5], 4'000'000);
  put64(&m[11], ref);
  put32(&m[19], shares);
  return m;
}

std::vector<uint8_t> make_delete(uint64_t ref) {
  std::vector<uint8_t> m(19, 0);
  m[0] = 'D';
  put16(&m[1], 1);
  put16(&m[3], 1);
  put48(&m[5], 5'000'000);
  put64(&m[11], ref);
  return m;
}

std::vector<uint8_t> make_replace(uint64_t old_ref, uint64_t new_ref, uint32_t shares,
                                  uint32_t price) {
  std::vector<uint8_t> m(35, 0);
  m[0] = 'U';
  put16(&m[1], 1);
  put16(&m[3], 1);
  put48(&m[5], 6'000'000);
  put64(&m[11], old_ref);
  put64(&m[19], new_ref);
  put32(&m[27], shares);
  put32(&m[31], price);
  return m;
}

int g_failures = 0;

void check(bool ok, const std::string& what) {
  if (ok) {
    std::cout << "  pass  " << what << "\n";
  } else {
    std::cout << "  FAIL  " << what << "\n";
    ++g_failures;
  }
}

void check_eq(uint64_t got, uint64_t want, const std::string& what) {
  if (got == want) {
    std::cout << "  pass  " << what << " = " << got << "\n";
  } else {
    std::cout << "  FAIL  " << what << ": got " << got << ", want " << want << "\n";
    ++g_failures;
  }
}

// Pull one field out of a CSV line.
std::string field(const std::string& line, size_t index) {
  size_t start = 0;
  for (size_t i = 0; i < index; ++i) {
    start = line.find(',', start);
    if (start == std::string::npos) return {};
    ++start;
  }
  const size_t end = line.find(',', start);
  return line.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

std::vector<std::string> data_lines(const std::string& text) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t nl = text.find('\n', pos);
    if (nl == std::string::npos) nl = text.size();
    const std::string line = text.substr(pos, nl - pos);
    if (!line.empty() && line[0] != '#' && line.rfind("seq,", 0) != 0 &&
        line.rfind("locate,", 0) != 0) {
      out.push_back(line);
    }
    pos = nl + 1;
  }
  return out;
}

int self_test() {
  std::cout << "ITCH 5.0 wire-format and book self-test\n";

  std::cout << "\nmessage sizes against the spec:\n";
  check_eq(message_size('A'), 36, "'A' Add Order");
  check_eq(message_size('F'), 40, "'F' Add Order with MPID");
  check_eq(message_size('E'), 31, "'E' Order Executed");
  check_eq(message_size('C'), 36, "'C' Order Executed with Price");
  check_eq(message_size('X'), 23, "'X' Order Cancel");
  check_eq(message_size('D'), 19, "'D' Order Delete");
  check_eq(message_size('U'), 35, "'U' Order Replace");
  check_eq(message_size('R'), 39, "'R' Stock Directory");
  check_eq(message_size('P'), 44, "'P' Trade");
  check_eq(message_size('Q'), 40, "'Q' Cross Trade");

  std::cout << "\nfield decode from golden bytes:\n";
  {
    const auto m = make_add(1, 3001, 'B', 300, "MSFT", 4'000'000);
    check_eq(m.size(), 36, "built 'A' length");
    check_eq(be16(&m[off::kLocate]), 1, "locate");
    check_eq(be48(&m[off::kTimestamp]), 2'000'000, "timestamp (6 bytes)");
    check_eq(be64(&m[off::kAddRef]), 3001, "order reference");
    check(m[off::kAddSide] == 'B', "side == 'B'");
    check_eq(be32(&m[off::kAddShares]), 300, "shares");
    check(std::memcmp(&m[off::kAddStock], "MSFT    ", 8) == 0, "stock == \"MSFT    \"");
    check_eq(be32(&m[off::kAddPrice]), 4'000'000, "price (raw)");
  }

  // Ken's original scenario: round lot 200, odd-lot bid 50 @ 400.00 above a
  // round-lot bid 300 @ 399.00. The round-lot best bid must be the 399.00 level.
  std::cout << "\nround-lot best, aggregate policy (round lot 200):\n";
  auto run_scenario = [](RoundLotPolicy policy, size_t levels,
                         const std::vector<std::vector<uint8_t>>& msgs,
                         std::string& events_out, std::string& snapshot_out, Stats& stats) {
    std::ostringstream events;
    BookBuilder::Options opt;
    opt.levels = levels;
    opt.rl_policy = policy;
    opt.publish_on_change = false;
    BookBuilder engine(opt, &events);
    uint64_t seq = 1;
    for (const auto& m : msgs) engine.on_message(m.data(), m.size(), seq++);
    std::ostringstream snap;
    engine.write_snapshot(snap);
    events_out = events.str();
    snapshot_out = snap.str();
    stats = engine.stats();
  };

  {
    std::vector<std::vector<uint8_t>> msgs;
    msgs.push_back(make_directory(1, "MSFT", 200));
    msgs.push_back(make_add(1, 3001, 'B', 50, "MSFT", 4'000'000));
    msgs.push_back(make_add(1, 3002, 'B', 300, "MSFT", 3'990'000));
    msgs.push_back(make_add(1, 3003, 'S', 400, "MSFT", 4'010'000));

    std::string ev, snap;
    Stats st;
    run_scenario(RoundLotPolicy::Aggregate, 3, msgs, ev, snap, st);
    const auto lines = data_lines(snap);
    check_eq(lines.size(), 1, "snapshot rows");
    if (!lines.empty()) {
      const std::string& l = lines[0];
      check(field(l, 1) == "MSFT", "symbol trimmed to \"MSFT\"");
      check_eq(std::stoull(field(l, 2)), 200, "round lot from 'R'");
      // columns: locate,symbol,round_lot,trading_state,bid_orders,ask_orders then view
      check_eq(std::stoull(field(l, 6)), 4'000'000, "bid_px_1 (odd lot on top)");
      check_eq(std::stoull(field(l, 7)), 50, "bid_sz_1");
      check_eq(std::stoull(field(l, 8)), 3'990'000, "bid_px_2");
      check_eq(std::stoull(field(l, 9)), 300, "bid_sz_2");
      const size_t rl = 6 + 2 * 3 + 2 * 3;  // after 3 bid + 3 ask levels
      check_eq(std::stoull(field(l, rl + 0)), 3'990'000, "round-lot best bid price");
      check_eq(std::stoull(field(l, rl + 1)), 300, "round-lot best bid size");
      check_eq(std::stoull(field(l, rl + 2)), 4'010'000, "round-lot best ask price");
      check_eq(std::stoull(field(l, rl + 3)), 400, "round-lot best ask size");
    }
    check_eq(st.messages, 4, "messages processed");
    check_eq(st.unknown_order_ref, 0, "unknown_order_ref");
    check_eq(st.missing_level, 0, "missing_level");
  }

  // Four 50-share odd lots at one price: 200 aggregate, but no single round lot.
  std::cout << "\naggregate vs order policy on stacked odd lots:\n";
  {
    std::vector<std::vector<uint8_t>> msgs;
    msgs.push_back(make_directory(1, "AAPL", 200));
    for (uint64_t i = 0; i < 4; ++i)
      msgs.push_back(make_add(1, 4000 + i, 'B', 50, "AAPL", 1'000'000));
    msgs.push_back(make_add(1, 4100, 'B', 250, "AAPL", 990'000));

    std::string ev, snap;
    Stats st;
    run_scenario(RoundLotPolicy::Aggregate, 2, msgs, ev, snap, st);
    auto lines = data_lines(snap);
    const size_t rl = 6 + 2 * 2 + 2 * 2;
    if (!lines.empty())
      check_eq(std::stoull(field(lines[0], rl)), 1'000'000,
               "aggregate policy takes the stacked odd lots");

    run_scenario(RoundLotPolicy::Order, 2, msgs, ev, snap, st);
    lines = data_lines(snap);
    if (!lines.empty()) {
      check_eq(std::stoull(field(lines[0], rl)), 990'000,
               "order policy skips them for the 250-share order");
      check_eq(std::stoull(field(lines[0], rl + 1)), 250, "order policy size");
    }
  }

  std::cout << "\norder lifecycle:\n";
  {
    std::vector<std::vector<uint8_t>> msgs;
    msgs.push_back(make_directory(1, "MSFT", 100));
    msgs.push_back(make_add(1, 1, 'B', 500, "MSFT", 1'000'000));
    msgs.push_back(make_exec(1, 200));                      // 500 -> 300
    msgs.push_back(make_cancel(1, 100));                    // 300 -> 200
    msgs.push_back(make_replace(1, 2, 400, 1'010'000));     // move and resize
    msgs.push_back(make_add(1, 3, 'S', 100, "MSFT", 1'020'000, true));  // 'F'
    msgs.push_back(make_delete(3));                         // ask gone

    std::string ev, snap;
    Stats st;
    run_scenario(RoundLotPolicy::Aggregate, 1, msgs, ev, snap, st);
    const auto lines = data_lines(snap);
    check_eq(lines.size(), 1, "snapshot rows");
    if (!lines.empty()) {
      check_eq(std::stoull(field(lines[0], 6)), 1'010'000, "bid price after replace");
      check_eq(std::stoull(field(lines[0], 7)), 400, "bid size after replace");
      check_eq(std::stoull(field(lines[0], 8)), 0, "ask absent after delete");
      check_eq(std::stoull(field(lines[0], 4)), 1, "bid order count");
      check_eq(std::stoull(field(lines[0], 5)), 0, "ask order count");
    }
    check_eq(st.by_type[static_cast<unsigned char>('F')], 1, "'F' messages seen");
    check_eq(st.unknown_order_ref, 0, "unknown_order_ref");
    check_eq(st.missing_level, 0, "missing_level");
    check_eq(st.exec_overfill, 0, "exec_overfill");
  }

  std::cout << "\nanomalies are counted, not crashed on:\n";
  {
    std::vector<std::vector<uint8_t>> msgs;
    msgs.push_back(make_directory(1, "MSFT", 100));
    msgs.push_back(make_add(1, 10, 'B', 100, "MSFT", 1'000'000));
    msgs.push_back(make_exec(10, 5000));        // overfill: would wrap if subtracted raw
    msgs.push_back(make_cancel(999, 100));      // unknown reference
    msgs.push_back(make_delete(999));           // unknown reference
    msgs.push_back(make_add(1, 11, 'B', 100, "MSFT", 1'000'000));
    msgs.push_back(make_add(1, 11, 'B', 200, "MSFT", 1'000'000));  // duplicate reference

    std::string ev, snap;
    Stats st;
    run_scenario(RoundLotPolicy::Aggregate, 1, msgs, ev, snap, st);
    check_eq(st.exec_overfill, 1, "exec_overfill");
    check_eq(st.unknown_order_ref, 2, "unknown_order_ref");
    check_eq(st.duplicate_order_ref, 1, "duplicate_order_ref");
    const auto lines = data_lines(snap);
    if (!lines.empty()) {
      // The overfilled order is gone; the duplicate leaves exactly 200 shares, not 300.
      check_eq(std::stoull(field(lines[0], 7)), 200, "bid size after duplicate replaced");
      check_eq(std::stoull(field(lines[0], 4)), 1, "bid order count");
    }
  }

  std::cout << "\ntruncated and unknown messages:\n";
  {
    std::ostringstream events;
    BookBuilder::Options opt;
    BookBuilder engine(opt, &events);
    const auto good = make_add(1, 1, 'B', 100, "MSFT", 1'000'000);
    engine.on_message(good.data(), 35, 1);  // one byte short of spec
    const uint8_t junk[4] = {'z', 0, 0, 0};
    engine.on_message(junk, sizeof(junk), 2);
    check_eq(engine.stats().truncated_message, 1, "truncated_message");
    check_eq(engine.stats().unknown_type, 1, "unknown_type");
    check_eq(engine.stats().events_emitted, 0, "events_emitted");
  }

  std::cout << "\n";
  if (g_failures == 0) {
    std::cout << "all self-tests passed\n";
    return 0;
  }
  std::cout << g_failures << " self-test failure(s)\n";
  return 1;
}

// ============================================================================
// Command line
// ============================================================================

void usage() {
  std::cout <<
      "usage: itch5_reference [options]\n"
      "  --in FILE            input capture or session file (default: stdin)\n"
      "  --format FMT         auto | pcap | binaryfile   (default auto)\n"
      "  --events FILE        per-event book stream, CSV (default: none)\n"
      "  --snapshot FILE      final book state for every symbol, CSV\n"
      "  --counters FILE      anomaly counters (default: stdout)\n"
      "  --levels N           book depth per side in the output (default 5)\n"
      "  --rl-policy P        aggregate | order  (default aggregate)\n"
      "                         aggregate: level qualifies when summed shares >= round lot\n"
      "                         order:     level qualifies when one order is >= round lot\n"
      "  --publish MODE       on-change | always   (default on-change)\n"
      "  --symbol SYM         restrict output to this symbol; repeatable\n"
      "  --max-messages N     stop after N ITCH messages\n"
      "  --self-test          run golden wire-format tests and exit\n"
      "  --help\n";
}

bool starts_with(std::string_view s, std::string_view prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string in_path, events_path, snapshot_path, counters_path, format = "auto";
  BookBuilder::Options opt;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    std::string val;
    bool inline_val = false;
    if (starts_with(a, "--")) {
      if (const size_t eq = a.find('='); eq != std::string::npos) {
        val = a.substr(eq + 1);
        a = a.substr(0, eq);
        inline_val = true;
      }
    }
    auto need = [&](const char* name) -> std::string {
      if (inline_val) return val;
      if (i + 1 >= argc) throw std::runtime_error(std::string(name) + " needs a value");
      return argv[++i];
    };

    if (a == "--help" || a == "-h") {
      usage();
      return 0;
    }
    if (a == "--self-test") return self_test();

    try {
      if (a == "--in") in_path = need("--in");
      else if (a == "--format") format = need("--format");
      else if (a == "--events") events_path = need("--events");
      else if (a == "--snapshot") snapshot_path = need("--snapshot");
      else if (a == "--counters") counters_path = need("--counters");
      else if (a == "--levels") opt.levels = std::stoul(need("--levels"));
      else if (a == "--rl-policy") {
        const std::string p = need("--rl-policy");
        if (p == "aggregate") opt.rl_policy = RoundLotPolicy::Aggregate;
        else if (p == "order") opt.rl_policy = RoundLotPolicy::Order;
        else throw std::runtime_error("--rl-policy must be aggregate or order");
      } else if (a == "--publish") {
        const std::string p = need("--publish");
        if (p == "on-change") opt.publish_on_change = true;
        else if (p == "always") opt.publish_on_change = false;
        else throw std::runtime_error("--publish must be on-change or always");
      } else if (a == "--symbol") {
        opt.symbol_filter.insert(need("--symbol"));
      } else if (a == "--max-messages") {
        opt.max_messages = std::stoull(need("--max-messages"));
      } else {
        std::cerr << "unknown option " << a << " (see --help)\n";
        return 2;
      }
    } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return 2;
    }
  }

  if (opt.levels == 0 || opt.levels > 100) {
    std::cerr << "error: --levels must be between 1 and 100\n";
    return 2;
  }
  if (format != "auto" && format != "pcap" && format != "binaryfile") {
    std::cerr << "error: --format must be auto, pcap or binaryfile\n";
    return 2;
  }

  try {
    std::ifstream file_in;
    std::istream* in = &std::cin;
    if (!in_path.empty()) {
      file_in.open(in_path, std::ios::binary);
      if (!file_in) throw std::runtime_error("cannot open input: " + in_path);
      in = &file_in;
    }

    // Auto-detect by pcap magic. Seekable input only; pipe in with an explicit --format.
    if (format == "auto") {
      uint8_t magic[4] = {};
      in->read(reinterpret_cast<char*>(magic), 4);
      const std::streamsize got = in->gcount();
      in->clear();
      if (!in->seekg(0)) {
        throw std::runtime_error("--format auto needs a seekable input; pass --format");
      }
      format = "binaryfile";
      if (got == 4) {
        const uint32_t m = be32(magic);
        if (m == 0xD4C3B2A1 || m == 0x4D3CB2A1 || m == 0xA1B2C3D4 || m == 0xA1B23C4D)
          format = "pcap";
      }
    }

    std::ofstream events_file;
    std::ostream* events = nullptr;
    if (!events_path.empty()) {
      events_file.open(events_path, std::ios::out | std::ios::trunc);
      if (!events_file) throw std::runtime_error("cannot open --events file: " + events_path);
      events = &events_file;
    }

    BookBuilder engine(opt, events);
    Stats input_stats{};

    if (format == "pcap") read_pcap(*in, engine, input_stats);
    else read_binaryfile(*in, engine);

    if (events) events_file.flush();

    if (!snapshot_path.empty()) {
      std::ofstream snap(snapshot_path, std::ios::out | std::ios::trunc);
      if (!snap) throw std::runtime_error("cannot open --snapshot file: " + snapshot_path);
      engine.write_snapshot(snap);
    }

    // pcap-level counters live outside the engine; fold them in for reporting.
    std::ostringstream counters;
    engine.write_stats(counters);
    if (format == "pcap") {
      counters << "pcap_packets_read\t" << input_stats.pcap_packets << "\n";
      counters << "pcap_non_udp_packets\t" << input_stats.non_udp_packets << "\n";
    }
    if (counters_path.empty()) {
      std::cout << counters.str();
    } else {
      std::ofstream cf(counters_path, std::ios::out | std::ios::trunc);
      if (!cf) throw std::runtime_error("cannot open --counters file: " + counters_path);
      cf << counters.str();
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
