#pragma once
// server.hpp — TCP client sessions: N symbols per client across any queues/feeds, replayed
// from a feed-time start in merged time order, then tailed live in the same loop.
//
// Request (one line): "<START_FEED_NS> <SEED 0|1> <SYM1> [SYM2 ...]\n"   (<= 256 symbols)
// Response: stream of OutQuote (64 B). With SEED=1 the last message before START for each
// symbol is sent first (prevailing state), oldest first.
//
// Ordering: a record with feed ts x is emitted once every LH the session touches has
// published a watermark >= x. So output is globally ordered by feed time, at the cost of
// waiting for the slowest LH (bounded by inter-packet time; SIP channels heartbeat).
// An LH whose watermark hasn't moved for kStaleNs (wall time) is dropped from the frontier
// until it moves again, so a dead or finished feed cannot stall a session. Ordering across
// LHs is then best-effort for the duration of the stall.
//
// One thread per client here. Production: epoll pool on non-isolated cores; the session
// body already never blocks on the store, only on send().

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

class OutBuf {
 public:
  explicit OutBuf(int fd) : fd_(fd) {}
  template <class T> void put(const T& v) noexcept {
    if (n_ + sizeof(T) > sizeof(buf_)) flush();
    std::memcpy(buf_ + n_, &v, sizeof(T));
    n_ += sizeof(T);
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

template <class L>
class ClientSession {
  using Cursor = QueueCursor<L>;
  static constexpr uint64_t kSeedMaxScan = uint64_t{1} << 26;
  static constexpr int64_t  kStaleNs     = 200'000'000;  // watermark staleness before an LH is ignored

 public:
  static void run(int fd, const Store<L>& st, std::stop_token stop) {
    char line[8192];
    std::size_t n = 0;
    while (n < sizeof(line) - 1) {
      const ssize_t r = ::recv(fd, line + n, sizeof(line) - 1 - n, 0);
      if (r <= 0) { ::close(fd); return; }
      n += static_cast<std::size_t>(r);
      if (std::memchr(line, '\n', n)) break;
    }
    const auto req = parse_request(std::string_view(line, n));
    if (!req) { ::close(fd); return; }
    Session s(fd, st, *req);
    s.loop(stop);
    ::close(fd);
  }

 private:
  struct Session {
    const Store<L>& st;
    Request req;
    std::unique_ptr<OutBuf> out;
    std::vector<SymKey> pending;              // symbols not yet seen by any LH
    std::vector<std::unique_ptr<Cursor>> cursors;
    std::vector<uint32_t> lhs;
    std::vector<std::pair<uint64_t, int64_t>> wm_seen;  // per lhs[i]: (last watermark, wall ns it changed)
    std::vector<std::pair<uint64_t, std::pair<Cursor*, uint64_t>>> seeds;  // (ts, (cursor, idx))

    Session(int fd, const Store<L>& s, Request r) : st(s), req(std::move(r)), out(std::make_unique<OutBuf>(fd)) {
      for (auto sv : req.syms) pending.push_back(make_key(sv.data(), static_cast<unsigned>(sv.size())));
      resolve();
    }

    Cursor& cursor_for(uint32_t gq) {
      for (auto& c : cursors) if (c->gq() == gq) return *c;
      auto& c = cursors.emplace_back(std::make_unique<Cursor>(st, gq));
      c->seek(c->find_start(req.start_ns));
      if (std::find(lhs.begin(), lhs.end(), c->lh()) == lhs.end()) { lhs.push_back(c->lh()); wm_seen.push_back({0, 0}); }
      return *c;
    }
    // Attach symbols that have appeared in SymQMap; called at start and whenever idle.
    void resolve() {
      for (std::size_t i = 0; i < pending.size();) {
        const auto e = st.symmap.find(pending[i]);
        if (!e) { ++i; continue; }
        Cursor& c = cursor_for(e->gq);
        c.add_symbol(e->symid);
        if (req.seed) if (const auto p = c.find_prior(e->symid, c.pos(), kSeedMaxScan)) seeds.push_back({c.ts_at(*p), {&c, *p}});
        pending[i] = pending.back(); pending.pop_back();
      }
    }
    void emit(const Cursor& c, uint64_t i) {
      c.visit(i, [&](const RecHdr& h, const uint8_t* m, uint64_t) {
        OutQuote o;
        if (const DecodeFn d = st.decoders[h.proto]; d && d(h, m, o)) out->put(o);
      });
    }
    // Emit everything with ts <= frontier, in ts order across cursors. Returns records emitted.
    std::size_t drain(uint64_t frontier) {
      std::size_t emitted = 0;
      for (;;) {
        Cursor* best = nullptr;
        uint64_t best_ts = UINT64_MAX;
        for (auto& c : cursors)
          if (const auto nx = c->peek(c->queue().committed()); nx && c->ts_at(*nx) <= frontier && c->ts_at(*nx) < best_ts) {
            best = c.get(); best_ts = c->ts_at(*nx);
          }
        if (!best) return emitted;
        emit(*best, *best->peek(best->queue().committed()));
        best->consume();
        ++emitted;
      }
    }
    void emit_seeds() {
      std::sort(seeds.begin(), seeds.end(), [](auto& a, auto& b) { return a.first < b.first; });
      for (auto& [ts, ci] : seeds) emit(*ci.first, ci.second);
      seeds.clear();
    }
    void loop(std::stop_token stop) {
      emit_seeds();
      Backoff bo;
      while (!stop.stop_requested() && out->ok()) {
        uint64_t frontier = UINT64_MAX;
        const int64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
        for (std::size_t i = 0; i < lhs.size(); ++i) {
          const uint64_t wm = st.time.watermark(lhs[i]);  // before committed()
          if (wm != wm_seen[i].first) wm_seen[i] = {wm, now};
          if (now - wm_seen[i].second < kStaleNs) frontier = std::min(frontier, wm);
        }
        if (drain(frontier) != 0) { out->flush(); bo.reset(); continue; }
        if (!pending.empty()) { resolve(); emit_seeds(); }
        bo.wait();
      }
      out->flush();
    }
  };
};

template <class L>
class Server {
 public:
  Server(const Store<L>& st, uint16_t port) : st_(st) {
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
                             ClientSession<L>::run(fd, st_, st);
                             done->store(true, std::memory_order_release);
                           }), done});
    }
  }
  const Store<L>& st_;
  int lfd_{-1};
  std::jthread acceptor_;
  std::mutex mu_;
  struct Slot { std::jthread t; std::shared_ptr<std::atomic<bool>> done; };
  std::vector<Slot> sessions_;
};

}  // namespace rawmd
