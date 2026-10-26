#pragma once
// server.hpp — TCP client sessions over the store: N symbols per client across any queues,
// from a feed-time start, in merged time order, then tailed live in the same loop.
//
// L1 request (one line): "<START_FEED_NS> <SEED 0|1> <SYM1> [SYM2 ...]\n"   (<= 256 symbols)
// L1 response: stream of OutQuote (64 B). With SEED=1, each symbol's latest message before
// its subscription start is sent first, flagged kQuoteSeed (prevailing state).
//
// SessionCore (below) is shared with the L3 session in l3_server.hpp. One thread per client
// here; production would run sessions on an epoll pool on non-isolated cores. A session
// never blocks on the store, only on send().

#include "rawmd.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <charconv>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string_view>
#include <sys/socket.h>
#include <unistd.h>

namespace rawmd {

inline int64_t steady_ns() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

class OutBuf {
 public:
  explicit OutBuf(int fd) : fd_(fd) {}
  template <class T> void put(const T& v) noexcept { put_bytes(&v, sizeof(T)); }
  void put_bytes(const void* p, std::size_t n) noexcept {
    if (n_ + n > sizeof(buf_)) flush();
    std::memcpy(buf_ + n_, p, n);
    n_ += n;
  }
  bool flush() noexcept {
    std::size_t off = 0;
    while (ok_ && off < n_) {
      const ssize_t r = ::send(fd_, buf_ + off, n_ - off, MSG_NOSIGNAL);
      if (r > 0) off += static_cast<std::size_t>(r);
      else if (r < 0 && errno == EINTR) continue;
      else ok_ = false;  // error or SO_SNDTIMEO: slow client dropped
    }
    n_ = 0;
    return ok_;
  }
  bool ok() const noexcept { return ok_; }
 private:
  int fd_;
  bool ok_{true};
  std::size_t n_{0};
  alignas(kCacheLine) std::byte buf_[64 * 1024];
};

// Reads the one-line request; nullopt on EOF, timeout or a line longer than the buffer.
inline std::optional<std::string> read_request_line(int fd) {
  std::string line(8192, '\0');
  std::size_t n = 0;
  while (n < line.size()) {
    const ssize_t r = ::recv(fd, line.data() + n, line.size() - n, 0);
    if (r <= 0) return std::nullopt;
    n += static_cast<std::size_t>(r);
    if (std::memchr(line.data(), '\n', n)) { line.resize(n); return line; }
  }
  return std::nullopt;
}

// ============================== SessionCore =================================
// Read-side machinery shared by the L1 and L3 sessions: one cursor per queue the session
// touches, the merge frontier, and per-subscription snapshots.
//
// Frontier: min watermark over the LHs the session reads; records are consumed in feed-time
// order up to it. An LH whose watermark hasn't moved for kStaleNs (wall time, counted from
// when the session started reading it) is left out of the min, so a quiet or dead feed
// cannot stall the session.
//
// Snapshots: each subscription has a snapshot time snap_ts. Its records with feed time
// < snap_ts go to H::on_silent; when the merge reaches snap_ts the session emits the
// subscription's snapshot (H::on_snapshot) and later records go to H::on_live. A symbol
// already known at connect gets snap_ts = START. One that first appears later gets
// snap_ts = max(START, F + 1), F = the latest feed time the session has consumed, and the
// records its queue's cursor has already passed are replayed to it (H::catch_up). Output
// therefore never goes back in time, and snapshot + later updates is complete.
//
// Degraded mode: if an LH left out as stale later delivers records older than output
// already sent, they still reach H::on_live (state stays right) but arrive after newer
// output from other LHs.
//
// H provides:
//   uint64_t origin(const Cursor&, uint64_t snap_ts)           first index for a new cursor
//   void on_attach(uint32_t sub, const SymKey&, const Cursor&)
//   void catch_up(uint32_t sub, const Cursor&, uint64_t end)    sub's records below end
//   void on_silent(uint32_t sub, const Cursor&, uint64_t i)
//   void on_live(uint32_t sub, const Cursor&, uint64_t i)
//   void on_snapshot(uint32_t sub, uint64_t snap_ts)
template <class L, class H>
class SessionCore {
 public:
  using Cursor = QueueCursor<L>;
  using Q = MsgQueue<L>;
  static constexpr int64_t  kStaleNs = 200'000'000;
  static constexpr uint32_t kNone    = UINT32_MAX;

  SessionCore(const Store<L>& st, H& h, uint64_t start_ns, int64_t resolve_ns)
      : st_(st), h_(h), start_(start_ns), resolve_ns_(resolve_ns), sub_of_(st.symmap.slots(), kNone) {}

  void subscribe(const SymKey& k) { pending_.push_back(k); }
  uint32_t symid(uint32_t sub) const noexcept { return subs_[sub].symid; }

  // One bounded unit of work; returns records consumed plus snapshots emitted.
  std::size_t step(std::size_t budget) {
    const int64_t now = steady_ns();
    if (!pending_.empty() && now - last_resolve_ >= resolve_ns_) { last_resolve_ = now; resolve(); }
    if (curs_.empty()) return 0;
    const uint64_t frontier = frontier_at(now);           // watermarks before committed()
    for (Cur& c : curs_) c.limit = c.c->queue().committed();
    std::size_t done = 0;
    // 1. Queues whose subscriptions are all still before their snapshot: nothing they hold
    //    below min_snap is published, so its order relative to other queues is moot.
    for (Cur& c : curs_) {
      if (c.n_live != 0) continue;
      const uint64_t hi = std::min(frontier, c.min_snap - 1);
      while (done < budget && fill(c) && c.ts <= hi) { take(c); ++done; }
    }
    // 2. Merge in feed-time order; a snapshot goes out before the first record at or past it.
    for (;;) {
      if (done >= budget) return done;
      Cur* best = nullptr;
      for (Cur& c : curs_) if (fill(c) && c.ts <= frontier && (!best || c.ts < best->ts)) best = &c;
      if (!best) break;
      if (best->ts >= next_snap_) done += snapshots(best->ts);
      take(*best);
      ++done;
    }
    // 3. Everything up to the frontier is consumed: snapshots due by frontier + 1 can go.
    return done + snapshots(frontier == UINT64_MAX ? UINT64_MAX : frontier + 1);
  }

 private:
  struct Sub { uint32_t symid, cur; uint64_t snap_ts; bool live; };
  struct Cur {
    std::unique_ptr<Cursor> c;
    uint32_t n_live{0};
    uint64_t min_snap{UINT64_MAX};   // over this queue's subscriptions not yet live
    uint64_t limit{0};               // committed() snapshot for this step
    bool has{false};                 // cached next match (idx, ts)
    uint64_t idx{0}, ts{0};
  };
  struct Lh { uint32_t lh; uint64_t wm; int64_t changed; };

  bool fill(Cur& c) {
    if (!c.has)
      if (const auto nx = c.c->peek(c.limit)) { c.has = true; c.idx = *nx; c.ts = c.c->ts_at(*nx); }
    return c.has;
  }
  void take(Cur& c) {
    const Q& q = c.c->queue();
    const uint32_t s = sub_of_[Q::sym_of(q.chunk(c.idx >> L::kShift))[c.idx & L::kMask]];
    if (subs_[s].live) h_.on_live(s, *c.c, c.idx);
    else h_.on_silent(s, *c.c, c.idx);
    consumed_ = std::max(consumed_, c.ts);
    c.c->consume();
    c.has = false;
  }
  uint64_t frontier_at(int64_t now) {
    uint64_t f = UINT64_MAX;
    for (Lh& l : lhs_) {
      const uint64_t wm = st_.time.watermark(l.lh);
      if (wm != l.wm) { l.wm = wm; l.changed = now; }
      if (now - l.changed < kStaleNs) f = std::min(f, wm);
    }
    return f;
  }
  void resolve() {
    for (std::size_t i = 0; i < pending_.size();) {
      if (const auto e = st_.symmap.find(pending_[i])) {
        attach(pending_[i], *e);
        pending_[i] = pending_.back();
        pending_.pop_back();
      } else {
        ++i;
      }
    }
  }
  void attach(const SymKey& key, SymQMap::Entry e) {
    if (sub_of_[e.symid] != kNone) return;                 // named twice in the request
    uint32_t ci = 0;
    while (ci < curs_.size() && curs_[ci].c->gq() != e.gq) ++ci;
    const bool fresh = ci == curs_.size();
    if (fresh) {
      curs_.emplace_back().c = std::make_unique<Cursor>(st_, e.gq);
      const uint32_t lh = curs_.back().c->lh();
      if (std::none_of(lhs_.begin(), lhs_.end(), [&](const Lh& l) { return l.lh == lh; }))
        lhs_.push_back({lh, st_.time.watermark(lh), steady_ns()});
    }
    Cur& c = curs_[ci];
    const uint64_t snap = std::max(start_, consumed_ + 1);
    const auto s = static_cast<uint32_t>(subs_.size());
    subs_.push_back({e.symid, ci, snap, false});
    sub_of_[e.symid] = s;
    h_.on_attach(s, key, *c.c);
    if (fresh) c.c->seek(h_.origin(*c.c, snap));
    h_.catch_up(s, *c.c, c.c->pos());                      // its records this cursor already passed
    c.c->add_symbol(e.symid);                              // later ones: rescan from pos()
    c.has = false;
    c.min_snap = std::min(c.min_snap, snap);
    next_snap_ = std::min(next_snap_, snap);
    building_.push_back(s);
  }
  std::size_t snapshots(uint64_t t) {
    if (t < next_snap_) return 0;
    std::stable_sort(building_.begin(), building_.end(), [&](uint32_t a, uint32_t b) { return subs_[a].snap_ts < subs_[b].snap_ts; });
    std::size_t k = 0;
    for (; k < building_.size() && subs_[building_[k]].snap_ts <= t; ++k) {
      Sub& s = subs_[building_[k]];
      h_.on_snapshot(building_[k], s.snap_ts);
      s.live = true;
      ++curs_[s.cur].n_live;
    }
    building_.erase(building_.begin(), building_.begin() + static_cast<std::ptrdiff_t>(k));
    next_snap_ = UINT64_MAX;
    for (Cur& c : curs_) c.min_snap = UINT64_MAX;
    for (const uint32_t b : building_) {
      const Sub& s = subs_[b];
      curs_[s.cur].min_snap = std::min(curs_[s.cur].min_snap, s.snap_ts);
      next_snap_ = std::min(next_snap_, s.snap_ts);
    }
    return k;
  }

  const Store<L>& st_;
  H& h_;
  const uint64_t start_;
  const int64_t resolve_ns_;
  int64_t last_resolve_{INT64_MIN / 2};
  uint64_t consumed_{0};               // F: latest feed time consumed
  uint64_t next_snap_{UINT64_MAX};     // earliest snap_ts among subscriptions not yet live
  std::vector<SymKey> pending_;        // requested, not yet in SymQMap
  std::vector<Sub> subs_;
  std::vector<uint32_t> sub_of_;       // symid -> subscription
  std::vector<uint32_t> building_;     // subscriptions before their snapshot
  std::vector<Cur> curs_;
  std::vector<Lh> lhs_;
};

// ============================ L1 client session ==============================
struct Request { uint64_t start_ns{0}; bool seed{false}; std::vector<std::string_view> syms; };

inline std::optional<Request> parse_request(std::string_view s) {
  auto next = [&](std::string_view& out) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    const auto e = s.find_first_of(" \t\r\n");
    out = s.substr(0, e);
    s.remove_prefix(e == std::string_view::npos ? s.size() : e);
    return !out.empty();
  };
  Request r;
  std::string_view t, sd, sym;
  if (!next(t) || std::from_chars(t.data(), t.data() + t.size(), r.start_ns).ec != std::errc{}) return std::nullopt;
  if (!next(sd)) return std::nullopt;
  r.seed = sd == "1";
  while (next(sym)) { if (sym.size() > 16 || r.syms.size() >= 256) return std::nullopt; r.syms.push_back(sym); }
  if (r.syms.empty()) return std::nullopt;
  return r;
}

// ResolveNs: how often symbols not yet seen by any LH are looked up again.
template <class L, int64_t ResolveNs = 1'000'000>
class ClientSession {
  using Cursor = QueueCursor<L>;
  using Q = MsgQueue<L>;
  static constexpr uint64_t    kSeedMaxScan = uint64_t{1} << 26;
  static constexpr uint64_t    kNoSeed      = UINT64_MAX;
  static constexpr std::size_t kBudget      = std::size_t{1} << 16;

 public:
  static void run(int fd, const Store<L>& st, std::stop_token stop) {
    const auto line = read_request_line(fd);
    const auto req = line ? parse_request(*line) : std::nullopt;
    if (req) {
      Session s(fd, st, *req);
      s.loop(stop);
    }
    ::close(fd);
  }

 private:
  struct Session {
    const Store<L>& st;
    Request req;
    std::unique_ptr<OutBuf> out;
    SessionCore<L, Session> core;
    std::vector<const Q*> q_of;           // per subscription
    std::vector<uint64_t> seed;           // per subscription: latest record before snap_ts

    Session(int fd, const Store<L>& s, Request r)
        : st(s), req(std::move(r)), out(std::make_unique<OutBuf>(fd)), core(s, *this, req.start_ns, ResolveNs) {
      for (auto sv : req.syms) core.subscribe(make_key(sv.data(), static_cast<unsigned>(sv.size())));
    }

    // ---- SessionCore hooks ----
    uint64_t origin(const Cursor& c, uint64_t snap_ts) const { return c.find_start(snap_ts); }
    void on_attach(uint32_t, const SymKey&, const Cursor& c) { q_of.push_back(&c.queue()); seed.push_back(kNoSeed); }
    void catch_up(uint32_t s, const Cursor& c, uint64_t end) {
      if (!req.seed) return;
      if (const auto p = c.find_prior(core.symid(s), end, kSeedMaxScan)) seed[s] = *p;
    }
    void on_silent(uint32_t s, const Cursor&, uint64_t i) { seed[s] = i; }
    void on_live(uint32_t, const Cursor& c, uint64_t i) { emit(c.queue(), i, 0); }
    void on_snapshot(uint32_t s, uint64_t) { if (req.seed && seed[s] != kNoSeed) emit(*q_of[s], seed[s], kQuoteSeed); }

    void emit(const Q& q, uint64_t i, uint8_t flags) {
      const std::byte* r = Q::rec_of(q.chunk(i >> L::kShift), i & L::kMask);
      const auto& h = *reinterpret_cast<const RecHdr*>(r);
      OutQuote o;
      if (const DecodeFn d = st.decoders[h.proto]; d && d(h, reinterpret_cast<const uint8_t*>(r + kRecHdr), o)) {
        o.flags = flags;
        out->put(o);
      }
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

// ================================== Server ===================================
template <class L, class SessionT = ClientSession<L>, class StoreT = Store<L>>
class Server {
 public:
  Server(const StoreT& st, uint16_t port) : st_(st) {
    lfd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (lfd_ < 0) throw std::runtime_error("socket");
    const int one = 1;
    ::setsockopt(lfd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (::bind(lfd_, reinterpret_cast<sockaddr*>(&a), sizeof(a)) < 0 || ::listen(lfd_, 64) < 0) throw std::runtime_error("bind/listen");
    acceptor_ = std::jthread([this](std::stop_token s) { accept_loop(s); });
  }
  ~Server() {
    acceptor_.request_stop();
    ::shutdown(lfd_, SHUT_RDWR);
    acceptor_.join();
    ::close(lfd_);
    std::lock_guard g(mu_);
    sessions_.clear();
  }
 private:
  void accept_loop(std::stop_token s) {
    while (!s.stop_requested()) {
      const int fd = ::accept(lfd_, nullptr, nullptr);
      if (fd < 0) { if (errno == EINTR) continue; break; }
      const int one = 1;
      ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      timeval tv{5, 0};
      ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
      ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
      std::lock_guard g(mu_);
      std::erase_if(sessions_, [](Slot& s) { return s.done->load(std::memory_order_acquire); });  // reap finished
      auto done = std::make_shared<std::atomic<bool>>(false);
      sessions_.push_back({std::jthread([fd, this, done](std::stop_token st) {
                             SessionT::run(fd, st_, st);
                             done->store(true, std::memory_order_release);
                           }), done});
    }
  }
  const StoreT& st_;
  int lfd_{-1};
  std::jthread acceptor_;
  std::mutex mu_;
  struct Slot { std::jthread t; std::shared_ptr<std::atomic<bool>> done; };
  std::vector<Slot> sessions_;
};

}  // namespace rawmd
