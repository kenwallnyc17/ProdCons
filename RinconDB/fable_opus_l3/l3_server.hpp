#pragma once
// l3_server.hpp — TCP client sessions for the L3 store.
//
// Request (one line): "<START_FEED_NS> <DEPTH 1..64> <SYM1> [SYM2 ...]\n"   (<= 256 symbols)
// Response, per symbol: one OutBookHdr kind 'S' + levels (the book after every message
// before the header's feed_ns), then OutBookHdr kind 'U' + levels after each later message
// that changes the top DEPTH levels on either side. Symbols known at connect snapshot at
// START; a symbol that first appears later snapshots at the session's position when it
// appears (>= START). Output is non-decreasing in feed time, history and live alike.
//
// Books are built from session open: each symbol's queue is scanned from index 0. Queues
// with no subscription past its snapshot yet are drained without the cross-queue merge.

#include "l3.hpp"
#include "server.hpp"

#include <array>

namespace rawmd::l3 {

struct Request3 { uint64_t start_ns{0}; uint32_t depth{20}; std::vector<std::string_view> syms; };

inline std::optional<Request3> parse_request3(std::string_view s) {
  auto next = [&](std::string_view& out) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    const auto e = s.find_first_of(" \t\r\n");
    out = s.substr(0, e);
    s.remove_prefix(e == std::string_view::npos ? s.size() : e);
    return !out.empty();
  };
  Request3 r;
  std::string_view t, d, sym;
  if (!next(t) || std::from_chars(t.data(), t.data() + t.size(), r.start_ns).ec != std::errc{}) return std::nullopt;
  if (!next(d) || std::from_chars(d.data(), d.data() + d.size(), r.depth).ec != std::errc{}) return std::nullopt;
  if (r.depth == 0 || r.depth > kMaxDepth) return std::nullopt;
  while (next(sym)) { if (sym.size() > 16 || r.syms.size() >= 256) return std::nullopt; r.syms.push_back(sym); }
  if (r.syms.empty()) return std::nullopt;
  return r;
}

// ResolveNs: how often symbols not yet seen by any LH are looked up again.
template <class Proto, class L, int64_t ResolveNs = 1'000'000>
class BookSession {
  using Cursor = QueueCursor<L>;
  using Q = MsgQueue<L>;
  static constexpr std::size_t kBudget = std::size_t{1} << 16;

 public:
  static void run(int fd, const Store<L>& st, std::stop_token stop) {
    const auto line = read_request_line(fd);
    const auto req = line ? parse_request3(*line) : std::nullopt;
    if (req) {
      Session s(fd, st, *req);
      s.loop(stop);
    }
    ::close(fd);
  }

 private:
  struct Sub {
    Sub(const SymKey& k, uint32_t id, uint32_t depth) : key(k), symid(id), book(depth) {}
    SymKey key;
    uint32_t symid;
    uint32_t seq{0};
    Book book;
    uint16_t nb{0}, na{0};                          // top N last sent
    std::array<Level, 2 * kMaxDepth> last;
  };

  struct Session {
    const Store<L>& st;
    Request3 req;
    std::unique_ptr<OutBuf> out;
    SessionCore<L, Session> core;
    std::vector<std::unique_ptr<Sub>> subs;
    alignas(64) std::byte msg[sizeof(OutBookHdr) + 2 * kMaxDepth * sizeof(Level)];

    Session(int fd, const Store<L>& s, Request3 r)
        : st(s), req(std::move(r)), out(std::make_unique<OutBuf>(fd)), core(s, *this, req.start_ns, ResolveNs) {
      for (auto sv : req.syms) core.subscribe(make_key(sv.data(), static_cast<unsigned>(sv.size())));
    }

    // ---- SessionCore hooks ----
    uint64_t origin(const Cursor&, uint64_t) const { return 0; }    // books start at session open
    void on_attach(uint32_t s, const SymKey& k, const Cursor&) { subs.push_back(std::make_unique<Sub>(k, core.symid(s), req.depth)); }
    void catch_up(uint32_t s, const Cursor& c, uint64_t end) {
      if (end == 0) return;
      Cursor scan(st, c.gq());
      scan.add_symbol(subs[s]->symid);
      while (const auto i = scan.peek(end)) { apply(*subs[s], c.queue(), *i); scan.consume(); }
    }
    void on_silent(uint32_t s, const Cursor& c, uint64_t i) { apply(*subs[s], c.queue(), i); }
    void on_live(uint32_t s, const Cursor& c, uint64_t i) {
      const Q& q = c.queue();
      const std::byte* ch = q.chunk(i >> L::kShift);
      const auto& h = *reinterpret_cast<const RecHdr*>(Q::rec_of(ch, i & L::kMask));
      if (apply(*subs[s], q, i)) publish(*subs[s], 'U', Q::ts_of(ch)[i & L::kMask], h.rx_ns, h.type);
    }
    void on_snapshot(uint32_t s, uint64_t snap_ts) { publish(*subs[s], 'S', snap_ts, 0, 0); }

    // True if the top N may have changed.
    bool apply(Sub& sub, const Q& q, uint64_t i) {
      const std::byte* r = Q::rec_of(q.chunk(i >> L::kShift), i & L::kMask);
      BookEvent ev;
      return Proto::decode(*reinterpret_cast<const RecHdr*>(r), reinterpret_cast<const uint8_t*>(r + kRecHdr), ev) && sub.book.apply(ev);
    }
    void publish(Sub& s, uint8_t kind, uint64_t feed_ns, uint64_t rx_ns, uint8_t trigger) {
      Level* lv = reinterpret_cast<Level*>(msg + sizeof(OutBookHdr));
      const uint32_t nb = s.book.top_bids(lv);
      const uint32_t na = s.book.top_asks(lv + nb);
      const std::size_t bytes = (nb + na) * sizeof(Level);
      if (kind == 'U' && nb == s.nb && na == s.na && std::memcmp(lv, s.last.data(), bytes) == 0) return;   // top N unchanged
      s.nb = static_cast<uint16_t>(nb);
      s.na = static_cast<uint16_t>(na);
      std::memcpy(s.last.data(), lv, bytes);
      auto* h = reinterpret_cast<OutBookHdr*>(msg);
      *h = OutBookHdr{feed_ns, rx_ns, {}, kind == 'S' ? 0u : ++s.seq, static_cast<uint16_t>(nb), static_cast<uint16_t>(na),
                      kind, Proto::kPxExp, trigger, s.book.consistent() ? uint8_t{0} : kFlagInconsistent, {}};
      std::memcpy(h->sym, &s.key, 16);
      out->put_bytes(msg, sizeof(OutBookHdr) + bytes);
    }
    void loop(std::stop_token stop) {
      Backoff bo;
      while (!stop.stop_requested() && out->ok()) {
        if (core.step(kBudget) != 0) { out->flush(); bo.reset(); }
        else bo.wait();
      }
      out->flush();
    }
  };
};

}  // namespace rawmd::l3
