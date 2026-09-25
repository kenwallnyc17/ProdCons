// main.cpp — self-test: 2 LHs fed synthetic ToyL1 packets, TCP server, and a client that
// connects mid-stream asking for history from an earlier start time, then tails live.
// Verifies the client stream against producer-side ground truth.
//
//   g++ -std=c++23 -O3 -march=sapphirerapids -pthread main.cpp -o rawmd && ./rawmd

#include "server.hpp"
#include "toy_l1.hpp"

#include <random>
#include <time.h>

using namespace rawmd;
using L  = Layout<128, 12>;  // small chunks for the test; DefaultLayout for production
using Q  = MsgQueue<L>;
using LH = LineHandler<ToyL1, Q>;

static uint64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return uint64_t(ts.tv_sec) * 1'000'000'000ull + uint64_t(ts.tv_nsec);
}

struct PktBuilder {
  alignas(64) uint8_t buf[1500];
  std::size_t n = 8;
  void reset(uint32_t seq) { std::memset(buf, 0, 8); std::memcpy(buf + 4, &seq, 4); n = 8; }
  void add(char type, const char* sym, uint64_t sip, int64_t bpx, int64_t apx, uint32_t bsz, uint32_t asz) {
    const uint16_t len = type == 'N' ? ToyL1::kNbboLen : ToyL1::kQuoteLen;
    uint8_t* m = buf + n;
    std::memset(m, 0, len);
    std::memcpy(m, &len, 2);
    m[2] = uint8_t(type); m[3] = 'P';
    std::memcpy(m + 8, &sip, 8);
    std::memset(m + 16, ' ', 11);
    std::memcpy(m + 16, sym, std::strlen(sym));
    std::memcpy(m + 32, &bpx, 8); std::memcpy(m + 40, &apx, 8);
    std::memcpy(m + 48, &bsz, 4); std::memcpy(m + 52, &asz, 4);
    if (type == 'N') { m[56] = 'Q'; m[57] = 'Z'; }
    n += len;
    ++buf[0];
  }
};

int main() {
  constexpr uint32_t kSyms = 400, kProbe = 7;
  constexpr uint16_t kPort = 9100;
  const uint64_t t0 = now_ns();

  Store<Q>::Config cfg{
      .n_lh = 2, .q_per_lh = 8, .pool_chunks = 128, .max_chunks_per_q = 512, .max_symbols = 4096,
      .session_start_ns = t0 - 1'000'000'000ull, .bucket_ns = 1'000'000'000ull, .n_buckets = 3600};
  Store<Q> store(cfg);
  std::printf("chunk=%zu KB records/chunk=%zu hugetlb=%d\n", L::kBytes >> 10, L::kCap, store.pool.hugetlb());

  LH lh0(0, store), lh1(1, store);
  LH* lhs[2] = {&lh0, &lh1};
  Server<ToyL1, Q> server(store, kPort);

  const uint64_t T = t0 + 500'000'000ull;  // client asks for history from here
  std::atomic<uint64_t> truth_after{0};
  char probe_sym[16];
  std::snprintf(probe_sym, sizeof probe_sym, "S%04u", kProbe);

  {
    std::vector<std::jthread> producers;
    for (uint32_t id = 0; id < 2; ++id)
      producers.emplace_back([&, id](std::stop_token st) {
        std::mt19937_64 rng(id + 1);
        PktBuilder pb;
        uint32_t seq = 0;
        char sym[16];
        while (!st.stop_requested()) {
          pb.reset(++seq);
          const uint64_t rx = now_ns();
          const int nm = 1 + int(rng() % 4);
          for (int i = 0; i < nm; ++i) {
            const uint32_t si = uint32_t(rng() % (kSyms / 2)) * 2 + id;  // channel partition
            std::snprintf(sym, sizeof sym, "S%04u", si);
            const int64_t mid = 100'0000 + int64_t(rng() % 5000);
            pb.add(rng() & 1 ? 'N' : 'Q', sym, rx - 1000, mid - 100, mid + 100,
                   100 * (1 + uint32_t(rng() % 9)), 100 * (1 + uint32_t(rng() % 9)));
            if (si == kProbe && rx >= T) truth_after.fetch_add(1, std::memory_order_relaxed);
          }
          lhs[id]->on_packet(pb.buf, pb.n, rx);
          for (int s = 0; s < 3000; ++s) cpu_relax();
        }
      });

    // Client connects 1.5 s in: ~1 s of replay, then live tail until producers stop.
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET; a.sin_port = htons(kPort); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) { std::perror("connect"); return 1; }
    timeval tv{1, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    char req[64];
    const int rl = std::snprintf(req, sizeof req, "%s %llu 1\n", probe_sym, (unsigned long long)T);
    ::send(fd, req, size_t(rl), 0);

    std::jthread stopper([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(1500));
      for (auto& p : producers) p.request_stop();
    });

    std::vector<OutQuote> got;
    alignas(64) uint8_t buf[1 << 16];
    std::size_t have = 0;
    for (;;) {
      const ssize_t r = ::recv(fd, buf + have, sizeof buf - have, 0);
      if (r <= 0) break;  // 1 s of silence after producers stop -> done
      have += size_t(r);
      const std::size_t whole = have / sizeof(OutQuote) * sizeof(OutQuote);
      for (std::size_t o = 0; o < whole; o += sizeof(OutQuote)) {
        OutQuote q; std::memcpy(&q, buf + o, sizeof q); got.push_back(q);
      }
      std::memmove(buf, buf + whole, have - whole);
      have -= whole;
    }
    ::close(fd);
    stopper.join();
    for (auto& p : producers) p.join();  // quiesce LHs before reading their stats

    uint64_t before = 0, after = 0, bad_sym = 0, non_mono = 0, prev = 0;
    for (const auto& q : got) {
      if (std::strncmp(q.sym, probe_sym, 16) != 0) ++bad_sym;
      if (q.rx_ns < prev) ++non_mono;
      prev = q.rx_ns;
      (q.rx_ns < T ? before : after)++;
    }
    std::printf("client: %zu msgs  seed(before T)=%llu  after T=%llu  truth=%llu  bad_sym=%llu  non_monotone=%llu\n",
                got.size(), (unsigned long long)before, (unsigned long long)after,
                (unsigned long long)truth_after.load(), (unsigned long long)bad_sym, (unsigned long long)non_mono);
    const bool pass = after == truth_after.load() && before <= 1 && bad_sym == 0 && non_mono == 0;
    for (uint32_t id = 0; id < 2; ++id) {
      const auto& s = lhs[id]->stats();
      std::printf("LH%u pkts=%llu msgs=%llu syms=%llu dropped=%llu\n", id, (unsigned long long)s.pkts,
                  (unsigned long long)s.msgs, (unsigned long long)s.new_symbols, (unsigned long long)s.dropped);
    }
    std::printf("pool: %zu/%zu arena chunks, %zu heap fallbacks\n%s\n", store.pool.arena_used(),
                store.pool.arena_chunks(), store.pool.heap_fallbacks(), pass ? "PASS" : "FAIL");
    return pass ? 0 : 2;
  }
}
