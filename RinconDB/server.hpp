#pragma once
// server.hpp — TCP client handling: replay from start time, then live tail, same loop.
//
// Request (one line):  "<SYMBOL> <START_NS> [SEED]\n"
//   SEED=1 also sends the last message for the symbol *before* START (prevailing BBO/NBBO).
// Response: stream of OutQuote (64 B each).
//
// One thread per client for clarity. Production: a small epoll-driven pool on non-isolated
// cores, each thread round-robining many sessions (the loop body below is already
// non-blocking with respect to the store; only send() blocks).

#include "rawmd.hpp"
#include "toy_l1.hpp"

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
  template <class T>
  void put(const T& v) noexcept {
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
      else ok_ = false;  // error, or SO_SNDTIMEO expired: slow client is dropped
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

struct Request {
  std::string_view sym;
  uint64_t start_ns{0};
  bool seed{false};
};

inline std::optional<Request> parse_request(std::string_view s) {
  auto next = [&](std::string_view& out) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    const auto e = s.find_first_of(" \t\r\n");
    out = s.substr(0, e);
    s.remove_prefix(e == std::string_view::npos ? s.size() : e);
    return !out.empty();
  };
  Request r;
  std::string_view t, sd;
  if (!next(r.sym) || r.sym.size() > 16 || !next(t)) return std::nullopt;
  if (std::from_chars(t.data(), t.data() + t.size(), r.start_ns).ec != std::errc{}) return std::nullopt;
  r.seed = next(sd) && sd == "1";
  return r;
}

template <class Proto, class Q>
struct ClientSession {
  static constexpr uint64_t kSeedMaxScan = uint64_t{1} << 26;  // backward search bound (records)
  static constexpr uint64_t kSliceRecs   = uint64_t{1} << 20;  // stop-token check granularity

  static void run(int fd, const Store<Q>& st, std::stop_token stop) {
    char line[256];
    std::size_t n = 0;
    while (n < sizeof(line) - 1) {
      const ssize_t r = ::recv(fd, line + n, sizeof(line) - 1 - n, 0);
      if (r <= 0) { ::close(fd); return; }
      n += static_cast<std::size_t>(r);
      if (std::memchr(line, '\n', n)) break;
    }
    const auto req = parse_request(std::string_view(line, n));
    if (!req) { ::close(fd); return; }

    const SymKey key = make_key(req->sym.data(), static_cast<unsigned>(req->sym.size()));
    Backoff bo;
    std::optional<SymQMap::Entry> e;
    while (!(e = st.symmap.find(key))) {  // symbol not seen yet (pre-open): wait for first message
      if (stop.stop_requested()) { ::close(fd); return; }
      bo.wait();
    }

    const Replayer<Q> rp(st, *e);
    const Q& q = rp.queue();
    const auto outp = std::make_unique<OutBuf>(fd);  // 64 KB: keep off the thread stack
    OutBuf& out = *outp;
    auto emit = [&](const RecHdr& h, const uint8_t* p, uint64_t ts) {
      OutQuote o;
      if (Proto::decode(p, h.len, h.type, ts, o)) out.put(o);
    };

    uint64_t cur = rp.find_start(req->start_ns);
    if (req->seed)
      if (const auto p = rp.find_prior(cur, kSeedMaxScan)) rp.visit(*p, emit);

    bo.reset();
    while (!stop.stop_requested() && out.ok()) {
      const uint64_t w = q.committed();
      if (cur < w) {
        const uint64_t to = std::min(w, cur + kSliceRecs);
        rp.scan(cur, to, emit);
        cur = to;
        out.flush();
        bo.reset();
      } else {
        bo.wait();  // caught up: live tail
      }
    }
    ::close(fd);
  }
};

template <class Proto, class Q>
class Server {
 public:
  Server(const Store<Q>& st, uint16_t port) : st_(st) {
    lfd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (lfd_ < 0) throw std::runtime_error("socket");
    const int one = 1;
    ::setsockopt(lfd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (::bind(lfd_, reinterpret_cast<sockaddr*>(&a), sizeof(a)) < 0 || ::listen(lfd_, 64) < 0)
      throw std::runtime_error("bind/listen");
    acceptor_ = std::jthread([this](std::stop_token s) { accept_loop(s); });
  }
  ~Server() {
    acceptor_.request_stop();
    ::shutdown(lfd_, SHUT_RDWR);
    acceptor_.join();
    ::close(lfd_);
    std::lock_guard g(mu_);
    sessions_.clear();  // jthread dtor: request_stop + join
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
      sessions_.emplace_back([fd, this](std::stop_token st) { ClientSession<Proto, Q>::run(fd, st_, st); });
    }
  }

  const Store<Q>& st_;
  int lfd_{-1};
  std::jthread acceptor_;
  std::mutex mu_;
  std::vector<std::jthread> sessions_;
};

}  // namespace rawmd
