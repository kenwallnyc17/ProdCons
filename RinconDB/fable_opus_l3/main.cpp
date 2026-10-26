// main.cpp — L1 self-test. Two LHs on a synthetic quote feed, 400 symbols; symbols 384..399
// only start quoting 2.0 s in, after the client has connected. The client asks for three
// active symbols, the sixteen late ones and one that never appears, from a feed-time start
// 1 s in the past, with seeds; then tails live. The session looks for unseen symbols only
// every 50 ms here, so late symbols attach after their first quotes. Checks, per symbol,
// record by record against producer ground truth:
//   * active symbols: the seed is the last quote before START; then every quote from START on;
//   * late symbols: an optional seed, then an unbroken run of quotes to the end, the seed
//     being the quote immediately before that run;
//   * all non-seed output is non-decreasing in feed time.
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

struct QRec {
  uint64_t feed_ns; int64_t bid, ask; uint32_t bsz, asz;
  bool operator==(const QRec&) const = default;
};
static QRec rec_of(const OutQuote& q) { return {q.feed_ns, q.bid_px, q.ask_px, q.bid_sz, q.ask_sz}; }
static std::string symname(uint32_t i) { char b[16]; std::snprintf(b, sizeof b, "S%04u", i); return b; }

int main(int argc, char** argv) {
  constexpr uint32_t kSyms = 400, kLate0 = 384;
  const std::vector<uint32_t> probes{7, 12, 31};   // 7,31 on LH1; 12 on LH0
  constexpr uint16_t kPort = 9100;
  const uint64_t t0 = now_ns();
  const uint64_t T = t0 + 500'000'000ull, t_late = t0 + 2'000'000'000ull;
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
  Server<L, ClientSession<L, 50'000'000>> server(store, kPort);

  std::vector<uint32_t> watch = probes;
  for (uint32_t s = kLate0; s < kSyms; ++s) watch.push_back(s);
  std::map<uint32_t, std::vector<QRec>> truth;     // keys fixed before the producers start; each written by one LH thread
  for (uint32_t s : watch) truth[s];

  std::vector<std::jthread> producers;
  for (uint32_t id = 0; id < 2; ++id)
    producers.emplace_back([&, id](std::stop_token st) {
      std::mt19937_64 rng(id + 1);
      PktBuilder pb;
      uint32_t seq = 0;
      while (!st.stop_requested()) {
        const uint32_t active = now_ns() >= t_late ? kSyms : kLate0;
        pb.reset(++seq);
        const int nm = 1 + int(rng() % 4);
        for (int i = 0; i < nm; ++i) {
          const uint32_t si = uint32_t(rng() % (active / 2)) * 2 + id;   // channel partition
          const uint64_t feed_ts = now_ns();                              // the feed's own clock
          const int64_t mid = 100'0000 + int64_t(rng() % 5000);
          const uint32_t bsz = 100 * (1 + uint32_t(rng() % 9)), asz = 100 * (1 + uint32_t(rng() % 9));
          pb.add(rng() & 1 ? 'N' : 'Q', symname(si).c_str(), feed_ts, mid - 100, mid + 100, bsz, asz);
          if (auto it = truth.find(si); it != truth.end()) it->second.push_back({feed_ts, mid - 100, mid + 100, bsz, asz});
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
  for (uint32_t s : watch) req += " " + symname(s);
  req += " ZZZZ\n";                                                    // never appears
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

  // ---- verify ----
  std::map<std::string, uint32_t> by_name;
  for (uint32_t s : watch) by_name[symname(s)] = s;
  std::map<uint32_t, std::vector<QRec>> live;
  std::map<uint32_t, std::vector<QRec>> seeds;
  uint64_t bad = 0, non_mono = 0, prev = 0;
  for (const auto& q : got) {
    const auto it = by_name.find(std::string(q.sym, strnlen(q.sym, 16)));
    if (it == by_name.end()) { ++bad; continue; }
    if (q.flags & kQuoteSeed) { seeds[it->second].push_back(rec_of(q)); continue; }
    if (q.feed_ns < prev) ++non_mono;
    prev = q.feed_ns;
    live[it->second].push_back(rec_of(q));
  }
  uint64_t wrong = 0, late_seeded = 0;
  for (uint32_t s : watch) {
    const auto& tr = truth[s];
    const auto& lv = live[s];
    const auto& sd = seeds[s];
    if (sd.size() > 1) { ++wrong; continue; }
    std::size_t j;                                                   // where the live run starts in truth
    if (s < kLate0) {
      j = std::size_t(std::lower_bound(tr.begin(), tr.end(), T, [](const QRec& r, uint64_t t) { return r.feed_ns < t; }) - tr.begin());
    } else {
      j = lv.empty() ? tr.size() : std::size_t(std::find(tr.begin(), tr.end(), lv.front()) - tr.begin());
      late_seeded += !sd.empty();
    }
    const bool run_ok = lv.size() == tr.size() - j && std::equal(lv.begin(), lv.end(), tr.begin() + std::ptrdiff_t(j));
    const bool seed_ok = j == 0 ? sd.empty() : (sd.size() == 1 && sd[0] == tr[j - 1]);
    if (!run_ok || !seed_ok) {
      ++wrong;
      std::printf("  %s: live %zu (truth from %zu: %zu) run_ok=%d seed_ok=%d\n", symname(s).c_str(), lv.size(), j, tr.size() - j, run_ok, seed_ok);
    }
  }
  const bool pass = bad == 0 && non_mono == 0 && wrong == 0;
  for (uint32_t p : probes)
    std::printf("%s: seed=%zu live=%zu truth>=START=%zu\n", symname(p).c_str(), seeds[p].size(), live[p].size(),
                std::size_t(std::count_if(truth[p].begin(), truth[p].end(), [&](const QRec& r) { return r.feed_ns >= T; })));
  std::printf("late symbols: %u checked, %llu joined with a seed (quotes before they attached)\n", kSyms - kLate0, (unsigned long long)late_seeded);
  std::printf("client: %zu msgs total, bad_sym=%llu non_monotone=%llu wrong=%llu\n", got.size(), (unsigned long long)bad,
              (unsigned long long)non_mono, (unsigned long long)wrong);
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
