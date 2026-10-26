// main_l3.cpp — L3 self-test and book micro-benchmark.
//
// 1. Unit tests (l3_tests.hpp): OpenTable clusters that fill the tail padding and overflow
//    into a grow, random churn; OrderRoute; LevelSide against std::map; Book on bad input.
// 2. End-to-end, for a locate-indexed feed (ITCH-shaped) and the same feed routed by order
//    id (PITCH-style): two LHs, 400 symbols; adds, deletes, cancels, executions, modifies,
//    replaces. Symbols 384..399 first appear (directory + orders) 2.0 s in, after the client
//    has connected. The client asks for 3 active symbols, the 16 late ones and one that never
//    appears, from a start 1 s in the past, depth 20. Here the session looks for unseen
//    symbols only every 50 ms, so late symbols attach well after their first messages, some
//    on a queue the session is already draining. Per symbol the test checks:
//      * exactly one snapshot, at START for symbols present at connect and >= START for
//        late ones, equal to the reference book before that time;
//      * updates exactly equal to the reference messages at/after it that changed the top
//        N, levels included;
//      * all output, snapshots included, non-decreasing in feed time; no inconsistency flags.
// 3. Benchmark: 5M synthetic events applied to one Book (~100K live orders).
//
//   g++ -std=c++23 -O3 -march=sapphirerapids -pthread main_l3.cpp -o rawmd_l3 && ./rawmd_l3
//   ./rawmd_l3 /path/on/nvme/pool.bin     (file backing)

#include "l3_server.hpp"
#include "l3_tests.hpp"
#include "toy_l3.hpp"

#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <time.h>

using namespace rawmd;
using namespace rawmd::l3;
using L  = Layout<12, ToyL3::kMaxMsg>;   // 4096 slots/chunk for the test; 16 in production
using St = Store<L>;

static uint64_t now_ns() {
  timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
  return uint64_t(ts.tv_sec) * 1'000'000'000ull + uint64_t(ts.tv_nsec);
}

// ---------------------------- reference book --------------------------------
struct RefOrder { uint8_t side; int64_t px; uint32_t qty; };
struct RefBook {
  std::map<int64_t, std::pair<uint32_t, uint32_t>, std::greater<>> bid;
  std::map<int64_t, std::pair<uint32_t, uint32_t>> ask;
  std::unordered_map<uint64_t, RefOrder> orders;
  std::vector<uint64_t> live;
  int64_t mid = 1'000'000;
  template <class M> static void add(M& m, int64_t px, uint32_t q) { auto& l = m[px]; l.first += q; l.second += 1; }
  template <class M> static void reduce(M& m, int64_t px, uint32_t q, bool gone) {
    auto it = m.find(px); it->second.first -= q; if (gone && --it->second.second == 0) m.erase(it);
  }
  void on_add(uint64_t id, uint8_t s, int64_t px, uint32_t q) { orders[id] = {s, px, q}; live.push_back(id); if (s == 'B') add(bid, px, q); else add(ask, px, q); }
  void on_reduce(uint64_t id, uint32_t q) { auto& o = orders[id]; o.qty -= q; const bool g = o.qty == 0; if (o.side == 'B') reduce(bid, o.px, q, g); else reduce(ask, o.px, q, g); if (g) orders.erase(id); }
  void on_delete(uint64_t id) { auto o = orders[id]; if (o.side == 'B') reduce(bid, o.px, o.qty, true); else reduce(ask, o.px, o.qty, true); orders.erase(id); }
  void on_modify(uint64_t id, int64_t px, uint32_t q) {
    const uint8_t s = orders[id].side;
    on_delete(id);
    orders[id] = {s, px, q};
    if (s == 'B') add(bid, px, q); else add(ask, px, q);
  }
  void top(unsigned n, std::vector<Level>& b, std::vector<Level>& a) const {
    b.clear(); a.clear();
    for (auto& [px, l] : bid) { if (b.size() >= n) break; b.push_back({px, l.first, l.second}); }
    for (auto& [px, l] : ask) { if (a.size() >= n) break; a.push_back({px, l.first, l.second}); }
  }
};
struct Truth { std::vector<Level> bid, ask; bool operator==(const Truth&) const = default; };

struct PktBuilder {
  alignas(64) uint8_t buf[1500];
  std::size_t n = 8;
  void reset(uint32_t seq) { std::memset(buf, 0, 8); std::memcpy(buf + 4, &seq, 4); n = 8; }
  uint8_t* hdr(uint16_t len, char type, uint16_t locate, uint64_t ts) {
    uint8_t* m = buf + n; std::memset(m, 0, len);
    std::memcpy(m, &len, 2); m[2] = uint8_t(type); std::memcpy(m + 4, &locate, 2); std::memcpy(m + 8, &ts, 8);
    n += len; ++buf[0]; return m;
  }
  bool fits(uint16_t len) const { return n + len <= sizeof buf && buf[0] < 255; }
};

static std::string symname(uint32_t i) { char b[16]; std::snprintf(b, sizeof b, "S%04u", i); return b; }

template <class Proto>
static bool run_e2e(const std::string& path, uint16_t port) {
  using LH = LineHandler3<Proto, L>;
  using Sess = BookSession<Proto, L, 50'000'000>;
  constexpr uint32_t kSyms = 400, kLate0 = 384, kDepth = 20;
  const std::vector<uint32_t> probes{7, 12, 31};
  std::vector<uint32_t> watch = probes;
  for (uint32_t s = kLate0; s < kSyms; ++s) watch.push_back(s);
  const std::set<uint32_t> watched(watch.begin(), watch.end());
  const uint64_t t0 = now_ns();
  const uint64_t T = t0 + 500'000'000ull, t_late = t0 + 2'000'000'000ull;

  St::Config cfg{.n_lh = 2, .q_per_lh = 8, .pool_chunks = 200, .max_chunks_per_q = 512, .max_symbols = 4096,
                 .session_start_ns = t0 - 1'000'000'000ull, .bucket_ns = 1'000'000'000ull, .n_buckets = 3600,
                 .backing = path.empty() ? ChunkPool::Backing::Ram : ChunkPool::Backing::File, .pool_path = path};
  St store(cfg);
  std::printf("e2e proto=%s backing=%s chunk=%zu KB slots/chunk=%zu stride=%zu B\n", Proto::kHasLocate ? "locate" : "order-id",
              path.empty() ? "ram" : "file", store.pool.chunk_bytes() >> 10, L::kCap, L::kRec);

  LH lh0(0, store, 20), lh1(1, store, 20);
  LH* lhs[2] = {&lh0, &lh1};
  Server<L, Sess> server(store, port);

  std::mutex truth_mu;
  std::map<uint32_t, std::map<uint64_t, Truth>> truth;   // watched sym -> feed_ts -> top N after that message

  std::vector<std::jthread> producers;
  for (uint32_t id = 0; id < 2; ++id)
    producers.emplace_back([&, id](std::stop_token stop) {
      std::mt19937_64 rng(id + 1);
      std::vector<RefBook> books(kSyms);
      std::vector<uint32_t> mine, late;
      for (uint32_t s = id; s < kSyms; s += 2) (s < kLate0 ? mine : late).push_back(s);
      PktBuilder pb;
      uint32_t seq = 0;
      uint64_t last_ts = 0, next_id = uint64_t{id + 1} << 40;
      auto ts_next = [&] { const uint64_t t = std::max(now_ns(), last_ts + 1); last_ts = t; return t; };
      auto directory = [&](const std::vector<uint32_t>& syms) {
        for (std::size_t i = 0; i < syms.size();) {
          pb.reset(++seq);
          while (i < syms.size() && pb.fits(24)) {
            uint8_t* m = pb.hdr(24, 'R', uint16_t(syms[i] + 1), ts_next());
            std::memset(m + 16, ' ', 8); const auto s = symname(syms[i]); std::memcpy(m + 16, s.data(), s.size());
            ++i;
          }
          lhs[id]->on_packet(pb.buf, pb.n, now_ns(), seq);
        }
      };
      directory(mine);
      bool late_on = false;
      std::vector<Level> tb, ta;
      while (!stop.stop_requested()) {
        if (!late_on && now_ns() >= t_late) { directory(late); mine.insert(mine.end(), late.begin(), late.end()); late_on = true; }
        pb.reset(++seq);
        const int nm = 1 + int(rng() % 6);
        for (int i = 0; i < nm; ++i) {
          const uint32_t si = mine[rng() % mine.size()];
          RefBook& rb = books[si];
          const uint16_t loc = uint16_t(si + 1);
          const uint64_t ts = ts_next();
          const unsigned r = unsigned(rng() % 100);
          auto rand_px = [&](uint8_t side) { const int64_t off = int64_t(rng() % 300) * 100; return side == 'B' ? rb.mid - 100 - off : rb.mid + 100 + off; };
          if (r < 3) { pb.hdr(17, 'S', 0, ts); continue; }                       // system event, not stored
          if (rb.live.empty() || r < 55) {
            const uint8_t side = rng() & 1 ? 'B' : 'S';
            const int64_t px = rand_px(side);
            const uint32_t q = 100 * (1 + uint32_t(rng() % 10));
            const uint64_t oid = ++next_id;
            uint8_t* m = pb.hdr(44, 'A', loc, ts);
            std::memcpy(m + 16, &oid, 8); m[24] = side; std::memcpy(m + 28, &q, 4);
            std::memset(m + 32, ' ', 8); const auto s = symname(si); std::memcpy(m + 32, s.data(), s.size());
            const uint32_t px32 = uint32_t(px); std::memcpy(m + 40, &px32, 4);
            rb.on_add(oid, side, px, q);
          } else {
            const std::size_t li = rng() % rb.live.size();
            const uint64_t oid = rb.live[li];
            const RefOrder o = rb.orders[oid];
            auto drop_live = [&] { rb.live[li] = rb.live.back(); rb.live.pop_back(); };
            if (r < 70) { uint8_t* m = pb.hdr(24, 'D', loc, ts); std::memcpy(m + 16, &oid, 8); rb.on_delete(oid); drop_live(); }
            else if (r < 82) {                                                   // partial cancel (or full via X)
              const uint32_t q = o.qty > 100 ? 100 : o.qty;
              uint8_t* m = pb.hdr(28, 'X', loc, ts); std::memcpy(m + 16, &oid, 8); std::memcpy(m + 24, &q, 4);
              rb.on_reduce(oid, q); if (q == o.qty) drop_live();
            } else if (r < 92) {                                                 // execution
              const uint32_t q = uint32_t(rng() % o.qty) + 1;
              uint8_t* m = pb.hdr(36, 'E', loc, ts); std::memcpy(m + 16, &oid, 8); std::memcpy(m + 24, &q, 4);
              rb.on_reduce(oid, q); if (q == o.qty) drop_live();
            } else if (r < 96) {                                                 // modify, same id
              const int64_t px = rand_px(o.side);
              const uint32_t q = 100 * (1 + uint32_t(rng() % 10));
              uint8_t* m = pb.hdr(32, 'M', loc, ts); std::memcpy(m + 16, &oid, 8); std::memcpy(m + 24, &q, 4);
              const uint32_t px32 = uint32_t(px); std::memcpy(m + 28, &px32, 4);
              rb.on_modify(oid, px, q);
            } else {                                                             // replace, new id
              const uint64_t nid = ++next_id;
              const int64_t px = rand_px(o.side);
              const uint32_t q = 100 * (1 + uint32_t(rng() % 10));
              uint8_t* m = pb.hdr(40, 'U', loc, ts); std::memcpy(m + 16, &oid, 8); std::memcpy(m + 24, &nid, 8);
              std::memcpy(m + 32, &q, 4); const uint32_t px32 = uint32_t(px); std::memcpy(m + 36, &px32, 4);
              rb.on_delete(oid); drop_live(); rb.on_add(nid, o.side, px, q);
            }
          }
          if (watched.count(si)) {
            rb.top(kDepth, tb, ta);
            std::lock_guard g(truth_mu);
            truth[si][ts] = Truth{tb, ta};
          }
        }
        lhs[id]->on_packet(pb.buf, pb.n, now_ns(), seq);
        for (int s = 0; s < 2000; ++s) cpu_relax();
      }
    });

  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) { std::perror("connect"); return false; }
  timeval tv{1, 0};
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  std::string req = std::to_string(T) + " " + std::to_string(kDepth);
  for (uint32_t s : watch) req += " " + symname(s);
  req += " ZZZZ\n";                                                              // never appears
  ::send(fd, req.data(), req.size(), 0);

  std::jthread stopper([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    for (auto& p : producers) p.request_stop();
  });

  struct Upd { OutBookHdr h; std::vector<Level> bid, ask; };
  std::vector<Upd> got;
  std::vector<uint8_t> buf;
  uint8_t tmp[1 << 16];
  for (;;) {
    const ssize_t r = ::recv(fd, tmp, sizeof tmp, 0);
    if (r <= 0) break;
    buf.insert(buf.end(), tmp, tmp + r);
    std::size_t o = 0;
    while (buf.size() - o >= sizeof(OutBookHdr)) {
      OutBookHdr h; std::memcpy(&h, buf.data() + o, sizeof h);
      const std::size_t need = sizeof h + (h.nbid + h.nask) * sizeof(Level);
      if (buf.size() - o < need) break;
      Upd u; u.h = h;
      const Level* lv = reinterpret_cast<const Level*>(buf.data() + o + sizeof h);
      u.bid.assign(lv, lv + h.nbid); u.ask.assign(lv + h.nbid, lv + h.nbid + h.nask);
      got.push_back(std::move(u));
      o += need;
    }
    buf.erase(buf.begin(), buf.begin() + std::ptrdiff_t(o));
  }
  ::close(fd);
  stopper.join();
  for (auto& p : producers) p.join();

  // ---- verify ----
  std::map<std::string, uint32_t> by_name;
  for (uint32_t s : watch) by_name[symname(s)] = s;
  std::map<uint32_t, std::vector<const Upd*>> per;
  uint64_t bad = 0, non_mono = 0, flagged = 0, prev = 0;
  for (const Upd& u : got) {
    const auto it = by_name.find(std::string(u.h.sym, strnlen(u.h.sym, 16)));
    if (it == by_name.end()) { ++bad; continue; }
    if (u.h.feed_ns < prev) ++non_mono;
    prev = u.h.feed_ns;
    flagged += u.h.flags != 0;
    per[it->second].push_back(&u);
  }
  uint64_t snaps = 0, updates = 0, expected = 0, mismatched = 0, late_after_T = 0;
  for (uint32_t s : watch) {
    const auto& v = per[s];
    const auto& tr = truth[s];
    if (v.empty() || v[0]->h.kind != 'S') { ++bad; std::printf("  %s: no snapshot first\n", symname(s).c_str()); continue; }
    ++snaps;
    const uint64_t S = v[0]->h.feed_ns;
    if (S < T || (s < kLate0 && S != T)) ++bad;
    late_after_T += s >= kLate0 && S > T;
    auto it = tr.lower_bound(S);                       // reference state before S
    Truth state = it == tr.begin() ? Truth{} : std::prev(it)->second;
    if (!(state == Truth{v[0]->bid, v[0]->ask})) { ++mismatched; std::printf("  %s: snapshot differs\n", symname(s).c_str()); }
    std::size_t k = 1;
    for (; it != tr.end(); ++it) {
      if (it->second == state) continue;
      state = it->second;
      ++expected;
      if (k < v.size() && v[k]->h.kind == 'U' && v[k]->h.feed_ns == it->first && Truth{v[k]->bid, v[k]->ask} == state) ++k;
      else ++mismatched;
    }
    if (k != v.size()) ++mismatched;                   // updates the reference doesn't have
    updates += v.size() - 1;
  }
  std::set<uint32_t> probe_gq;
  uint32_t shared = 0;
  for (uint32_t p : probes) if (const auto e = store.symmap.find(make_key(symname(p).data(), 5))) probe_gq.insert(e->gq);
  for (uint32_t s = kLate0; s < kSyms; ++s) if (const auto e = store.symmap.find(make_key(symname(s).data(), 5)); e && probe_gq.count(e->gq)) ++shared;

  const bool pass = bad == 0 && mismatched == 0 && non_mono == 0 && flagged == 0 && snaps == watch.size() && updates == expected;
  std::printf("  client: snapshots=%llu/%zu (late after START: %llu) updates=%llu expected=%llu mismatched=%llu bad=%llu non_monotone=%llu flagged=%llu\n",
              (unsigned long long)snaps, watch.size(), (unsigned long long)late_after_T, (unsigned long long)updates,
              (unsigned long long)expected, (unsigned long long)mismatched, (unsigned long long)bad,
              (unsigned long long)non_mono, (unsigned long long)flagged);
  std::printf("  late symbols on a queue the session was already reading: %u of %u\n", shared, kSyms - kLate0);
  for (uint32_t id = 0; id < 2; ++id) {
    const auto& s = lhs[id]->stats();
    std::printf("  LH%u pkts=%llu msgs=%llu syms=%llu dropped=%llu unknown_locate=%llu unknown_order=%llu dup=%llu route_full=%llu\n", id,
                (unsigned long long)s.pkts, (unsigned long long)s.msgs, (unsigned long long)s.new_symbols,
                (unsigned long long)(s.dropped + s.oversize), (unsigned long long)s.unknown_locate,
                (unsigned long long)s.unknown_order, (unsigned long long)s.dup_order, (unsigned long long)s.route_full);
  }
  std::printf("  pool: %zu/%zu arena chunks, %zu heap fallbacks\n  %s\n", store.pool.arena_used(), store.pool.arena_chunks(),
              store.pool.heap_fallbacks(), pass ? "PASS" : "FAIL");
  return pass;
}

static void bench() {
  constexpr uint32_t kDepth = 20;
  std::mt19937_64 rng(42);
  std::vector<BookEvent> evs; evs.reserve(5'000'000);
  std::vector<uint64_t> live; std::unordered_map<uint64_t, std::pair<uint8_t, uint32_t>> o;
  uint64_t next_id = 1;
  while (evs.size() < 5'000'000) {
    const unsigned r = unsigned(rng() % 100);
    if (live.empty() || (r < 55 && live.size() < 100'000)) {   // ~100K live orders: a busy symbol
      const uint8_t side = rng() & 1 ? 'B' : 'S';
      const int64_t off = int64_t(rng() % 2000) * 100;
      const uint32_t q = 100 * (1 + uint32_t(rng() % 10));
      evs.push_back({Ev::Add, side, q, side == 'B' ? 1'000'000 - 100 - off : 1'000'000 + 100 + off, next_id, 0});
      o[next_id] = {side, q}; live.push_back(next_id++);
    } else {
      const std::size_t li = rng() % live.size(); const uint64_t id = live[li];
      auto drop = [&] { live[li] = live.back(); live.pop_back(); o.erase(id); };
      if (r < 75) { evs.push_back({Ev::Delete, 0, 0, 0, id, 0}); drop(); }
      else if (r < 92) { auto& e = o[id]; const uint32_t q = uint32_t(rng() % e.second) + 1; evs.push_back({Ev::Reduce, 0, q, 0, id, 0}); e.second -= q; if (!e.second) drop(); }
      else { const uint8_t side = o[id].first; const int64_t off = int64_t(rng() % 2000) * 100; const uint32_t q = 100 * (1 + uint32_t(rng() % 10));
             evs.push_back({Ev::Replace, 0, q, side == 'B' ? 1'000'000 - 100 - off : 1'000'000 + 100 + off, id, next_id}); drop(); o[next_id] = {side, q}; live.push_back(next_id++); }
    }
  }
  Book b(kDepth);
  uint64_t changed = 0;
  const uint64_t t1 = now_ns();
  for (const auto& e : evs) changed += b.apply(e);
  const uint64_t t2 = now_ns();
  std::printf("bench: %zu events in %.1f ms = %.1f ns/event; top-%u changed %llu times; orders=%zu levels=%zu/%zu\n",
              evs.size(), (t2 - t1) / 1e6, double(t2 - t1) / evs.size(), kDepth, (unsigned long long)changed,
              b.orders(), b.bid_levels(), b.ask_levels());
}

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "";
  bool pass = test::run_units();
  pass &= run_e2e<ToyL3>(path, 9101);
  pass &= run_e2e<ToyL3NoLocate>(path.empty() ? "" : path + ".2", 9102);
  bench();
  std::puts(pass ? "ALL PASS" : "FAIL");
  return pass ? 0 : 2;
}
