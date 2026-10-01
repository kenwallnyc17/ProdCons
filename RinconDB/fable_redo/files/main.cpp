// main.cpp — self-test. Two LHs on a synthetic feed; one client subscribes to three symbols
// spread across both LHs, from a feed-time start 1 s in the past, with seeds; then tails live.
// Checks per-symbol counts against producer ground truth and global feed-time ordering.
//
//   g++ -std=c++23 -O3 -march=sapphirerapids -pthread main.cpp -o rawmd
//   ./rawmd            (RAM backing)      ./rawmd /path/on/nvme/pool.bin   (file backing)

#include "server.hpp"
#include "toy_l1.hpp"

#include <map>
#include <random>
#include <string>
#include <time.h>

using namespace rawmd;
using L  = Layout<12, ToyL1::kNbboLen>;  // 4096 slots/chunk for the test; 16 (64K) in production
using St = Store<L>;
using LH = LineHandler<ToyL1, L>;

static uint64_t now_ns() {
  timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
  return uint64_t(ts.tv_sec) * 1'000'000'000ull + uint64_t(ts.tv_nsec);
}

struct PktBuilder {
  alignas(64) uint8_t buf[1500];
  std::size_t n = 8;
  void reset(uint32_t seq) { std::memset(buf, 0, 8); std::memcpy(buf + 4, &seq, 4); n = 8; }
  void add(char type, const char* sym, uint64_t feed_ts, int64_t bpx, int64_t apx, uint32_t bsz, uint32_t asz) {
    const uint16_t len = type == 'N' ? ToyL1::kNbboLen : ToyL1::kQuoteLen;
    uint8_t* m = buf + n;
    std::memset(m, 0, len);
    std::memcpy(m, &len, 2); m[2] = uint8_t(type); m[3] = 'P';
    std::memcpy(m + 8, &feed_ts, 8);
    std::memset(m + 16, ' ', 11); std::memcpy(m + 16, sym, std::strlen(sym));
    std::memcpy(m + 32, &bpx, 8); std::memcpy(m + 40, &apx, 8);
    std::memcpy(m + 48, &bsz, 4); std::memcpy(m + 52, &asz, 4);
    if (type == 'N') { m[56] = 'Q'; m[57] = 'Z'; }
    n += len; ++buf[0];
  }
};

int main(int argc, char** argv) {
  constexpr uint32_t kSyms = 400;
  const std::vector<uint32_t> probes{7, 12, 31};   // 7,31 on LH1; 12 on LH0
  constexpr uint16_t kPort = 9100;
  const uint64_t t0 = now_ns();
  const std::string path = argc > 1 ? argv[1] : "";

  St::Config cfg{.n_lh = 2, .q_per_lh = 8, .pool_chunks = 96,
                 .max_chunks_per_q = 512, .max_symbols = 4096, .session_start_ns = t0 - 1'000'000'000ull,
                 .bucket_ns = 1'000'000'000ull, .n_buckets = 3600,
                 .backing = path.empty() ? ChunkPool::Backing::Ram : ChunkPool::Backing::File, .pool_path = path};
  St store(cfg);
  std::printf("backing=%s chunk=%zu KB slots/chunk=%zu stride=%zu B (max msg %zu)\n", path.empty() ? "ram" : "file",
              store.pool.chunk_bytes() >> 10, L::kCap, L::kRec, L::kMaxMsg);

  LH lh0(0, store), lh1(1, store);
  LH* lhs[2] = {&lh0, &lh1};
  Server<L> server(store, kPort);

  const uint64_t T = t0 + 500'000'000ull;
  std::map<uint32_t, std::atomic<uint64_t>> truth;
  for (uint32_t p : probes) truth[p] = 0;
  auto symname = [](uint32_t i) { char b[16]; std::snprintf(b, sizeof b, "S%04u", i); return std::string(b); };

  std::vector<std::jthread> producers;
  for (uint32_t id = 0; id < 2; ++id)
    producers.emplace_back([&, id](std::stop_token st) {
      std::mt19937_64 rng(id + 1);
      PktBuilder pb;
      uint32_t seq = 0;
      while (!st.stop_requested()) {
        pb.reset(++seq);
        const int nm = 1 + int(rng() % 4);
        for (int i = 0; i < nm; ++i) {
          const uint32_t si = uint32_t(rng() % (kSyms / 2)) * 2 + id;   // channel partition
          const uint64_t feed_ts = now_ns();                            // the feed's own clock
          const int64_t mid = 100'0000 + int64_t(rng() % 5000);
          pb.add(rng() & 1 ? 'N' : 'Q', symname(si).c_str(), feed_ts, mid - 100, mid + 100,
                 100 * (1 + uint32_t(rng() % 9)), 100 * (1 + uint32_t(rng() % 9)));
          if (auto it = truth.find(si); it != truth.end() && feed_ts >= T) it->second.fetch_add(1, std::memory_order_relaxed);
        }
        lhs[id]->on_packet(pb.buf, pb.n, now_ns(), seq);
        for (int s = 0; s < 3000; ++s) cpu_relax();
      }
    });

  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET; a.sin_port = htons(kPort); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) { std::perror("connect"); return 1; }
  timeval tv{1, 0};
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  std::string req = std::to_string(T) + " 1";
  for (uint32_t p : probes) req += " " + symname(p);
  req += "\n";
  ::send(fd, req.data(), req.size(), 0);

  std::jthread stopper([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    for (auto& p : producers) p.request_stop();
  });

  std::vector<OutQuote> got;
  alignas(64) uint8_t buf[1 << 16];
  std::size_t have = 0;
  for (;;) {
    const ssize_t r = ::recv(fd, buf + have, sizeof buf - have, 0);
    if (r <= 0) break;
    have += size_t(r);
    const std::size_t whole = have / sizeof(OutQuote) * sizeof(OutQuote);
    for (std::size_t o = 0; o < whole; o += sizeof(OutQuote)) { OutQuote q; std::memcpy(&q, buf + o, sizeof q); got.push_back(q); }
    std::memmove(buf, buf + whole, have - whole);
    have -= whole;
  }
  ::close(fd);
  stopper.join();
  for (auto& p : producers) p.join();

  std::map<std::string, uint64_t> seed_cnt, after_cnt;
  uint64_t bad_sym = 0, non_mono = 0, prev = 0;
  bool in_seeds = true;
  for (const auto& q : got) {
    std::string s(q.sym, strnlen(q.sym, 16));
    bool known = false;
    for (uint32_t p : probes) known |= s == symname(p);
    if (!known) { ++bad_sym; continue; }
    if (q.feed_ns < T && in_seeds) { ++seed_cnt[s]; continue; }
    in_seeds = false;
    if (q.feed_ns < prev) ++non_mono;
    prev = q.feed_ns;
    ++after_cnt[s];
  }
  bool pass = bad_sym == 0 && non_mono == 0;
  for (uint32_t p : probes) {
    const auto s = symname(p);
    std::printf("%s: seed=%llu after=%llu truth=%llu\n", s.c_str(), (unsigned long long)seed_cnt[s],
                (unsigned long long)after_cnt[s], (unsigned long long)truth[p].load());
    pass = pass && after_cnt[s] == truth[p].load() && seed_cnt[s] <= 1;
  }
  std::printf("client: %zu msgs total, bad_sym=%llu non_monotone=%llu\n", got.size(), (unsigned long long)bad_sym, (unsigned long long)non_mono);
  for (uint32_t id = 0; id < 2; ++id) {
    const auto& s = lhs[id]->stats();
    uint64_t late = 0, drop = 0;
    for (uint32_t k = 0; k < cfg.q_per_lh; ++k) { const auto& qs = store.queues[id * cfg.q_per_lh + k].stats(); late += qs.late_chunks; drop += qs.dropped + qs.oversize; }
    drop += s.oversize;
    std::printf("LH%u pkts=%llu msgs=%llu syms=%llu late_chunks=%llu dropped=%llu\n", id,
                (unsigned long long)s.pkts, (unsigned long long)s.msgs, (unsigned long long)s.new_symbols,
                (unsigned long long)late, (unsigned long long)(drop + s.dropped));
  }
  std::printf("pool: %zu/%zu arena chunks, %zu heap fallbacks, %zu mlock failures\n%s\n", store.pool.arena_used(),
              store.pool.arena_chunks(), store.pool.heap_fallbacks(), store.pool.mlock_failures(), pass ? "PASS" : "FAIL");
  return pass ? 0 : 2;
}
