// mdsys_2_cancel_bench.cpp — latency benchmark for md::CancelTactic (mdsys_2.hpp [4]).
//
//  part 1  steady-state cost per on_book_update, nothing fires: NBBO moved
//          (bid toggles), NBBO unchanged, execution (counts toward (d),
//          never reaches the threshold). Rows: an untracked symbol (no
//          orders, not pinned: the one-bit-test path), a pinned symbol with
//          no orders (NBBO read + estimators), and K = 1/4/8 orders per side;
//          "typical" (orders rejected at the cheap floors) and "full-path"
//          (every order reaches the lazy vol test).
//  part 2  decision latency: call entry -> callback entry, for a (d) cancel
//          on an execution and a (c) cancel on a non-execution update;
//          each trigger follows 15 non-firing updates of the same kind, so
//          branch predictors see the realistic "no fire" history. Hot, cold,
//          and cold with the decode-stage prefetch ("+pf", "+pfq": part 3).
//  part 3  part 1's typical rows without and with the decode-stage
//          prefetch: "+pf", CancelTactic::prefetch(sym) for the symbol two
//          updates ahead, issued inside the timed section, so the cell is
//          what the update costs with it, the prefetch's own cost included;
//          "+pfq", the same plus the host's prefetch of its NBBO record,
//          which the tactic cannot issue. An untracked symbol and K = 1/8,
//          hot; K = 1/8 cold. (Part 2 has the cold decisions with both.)
//  part 5/6/7  (d) cold only / (c) cold only / hot decisions only.
//  part 8  the round-lot hook body suggested in mdsys_2.hpp (two
//          MarketDataSystem::roundlot_best walks), hot: what the venue
//          volatility build adds to each tracked update, after the decision.
//  part 9  JoinTactic, through one call site per entry point (a function
//          each, as in a feed handler; the cells include that call):
//          unarmed symbol, Adds that do not signal (behind the
//          best; an odd lot; a round lot while the side's signal lives), an
//          Add that signals (the previous signal's kill flag read first; the
//          far-touch hook record in cache), executions with no live signal,
//          with the bid side's signal live, with both sides' (counts and tape
//          inserts), and one that reads a kill flag; hot, and cold (the
//          symbol's join and tape lines and far-touch record demoted to L3
//          after each call with CLDEMOTE), each also with JoinTactic::
//          prefetch(sym) for the symbol two messages ahead ("+pf").
//  part 10 CancelTactic (a)/(b): on_place with the queue walk (1, 8, 15, 40
//          orders ahead) vs (a)/(b) off, and with a live join signal to seed
//          (d) from; on_book_update for a removal at our price (b armed), at
//          another price, with no queue flags, and for our order reduced
//          ((a) re-checked); and, once an order's ratio is met, a removal at
//          its price and the cut of a flagged order there, which then read
//          the sizes ahead of it (one round lot left 4 / 15 / 40 / 200
//          orders ahead of ours, odd lots between).
//  part 11 part 1's typical rows in the default configuration's usual state:
//          "+gone", the removal messages carry kQGone as the book's remove()
//          reports it; "+armed", each order also rests behind a round-lot
//          order, so (b) is armed for it. Half the executions are at our
//          best order's price.
//  part 12 (b) and (a) decisions, call entry -> callback entry: the removal
//          that puts our order at the front; the same with an odd lot left
//          ahead of it; the cut that leaves the last round lot ahead an odd
//          lot; three orders of ours at the price decided together; (a) when
//          our order is cut.
//  part 13 part 2's decisions with (b) armed and kQGone on the removals.
//  part 15 on_place_late at the exchange's ack, beside on_place for the
//          same order: alone at its price, behind four round lots, behind
//          one of ours that had counted it, and with its own join signal.
//  part 14 one handler function calling both tactics, one call site per
//          entry point — a feed handler's shape, and the one in which the
//          compiler can inline the passes into the handler: per message, for
//          a symbol with no orders of ours, a pinned one, one order, one
//          order with (b) armed (messages at another price / at ours).
//          mdsys_2_host_bench.cpp has the same rows with the handler in a
//          translation unit of its own, which is what a host looks like to
//          the inliner.
// getNBBO here reads one 16-byte pair from a 64-byte record per symbol and is
// defined in this file, so it can inline: it stands in for the host's
// snapshot read with the record in cache. A record the SIP process has just
// rewritten on another core costs a cross-core transfer on top.
// Our orders sit in QBlockLists of a shared pool, one level each, so
// on_place flags real queue slots. In parts 1-9, (a) and (b) are on (the
// defaults) but each order is alone in its level and no measured update
// carries queue flags, and the (d) gate is off, so no measured path reads
// the queues; parts 10-14 rest orders behind others and pass the flags the
// book's remove() and reduce() return, as a host does.
// hot  = 8 symbols round-robin (state in L1/L2);
// cold = 8192 symbols in random order (state in L3; no stride for the
//        hardware prefetchers).
// Cells are TSC ticks (lfence;rdtsc;lfence .. rdtscp;lfence, timer overhead
// subtracted). "To callback" is lfence-serialized, i.e. an upper bound that
// includes everything sequenced before the callback.
//
// Build:                   g++ -std=c++23 -O3 -march=graniterapids -DNDEBUG -I. mdsys_2_cancel_bench.cpp -o bench
//   scalar reference build: add -DUSE_REF
//   volatility from the venue's round-lot BBO: add -DMD_CANCEL_VOL_FROM_VENUE=1 (here a
//   second 16-byte record that moves with the NBBO, i.e. a venue at the inside, so both
//   builds see the same events; it stands in for getVenueRoundLotBBO, whose real body
//   part 8 times on its own)
// Run pinned, huge pages for the tactic's arrays (glibc >= 2.35):
//   GLIBC_TUNABLES=glibc.malloc.hugetlb=1 taskset -c <isolated cpu> ./bench <label> <part>
#include "mdsys_2.hpp"

#include <x86intrin.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#ifdef USE_REF
using CT = md::CancelTacticT<false>;
#else
using CT = md::CancelTactic;
#endif
using namespace md;

// ---- the host's NBBO snapshot, bench version ----------------------------------

struct alignas(64) NbboRec { Nbbo q; char pad[48]; };
static std::vector<NbboRec> g_rec, g_vrec;
Nbbo md::getNBBO(uint16_t s) noexcept { return g_rec[s].q; }
Nbbo md::getVenueRoundLotBBO(uint16_t s) noexcept { return g_vrec[s].q; }
static void set_nbbo(uint16_t s, PriceType b, PriceType a) { g_rec[s].q = {b, a}; g_vrec[s].q = {b, a}; }
static std::vector<SymbolState> g_st(1u << 14);              // SymbolState per symbol (round lot, join state)
SymbolState& md::getSymbolState(uint16_t s) noexcept { return g_st[s]; }
static std::vector<RlQuote> g_rlq(2u << 14);                 // join far touch: one record per symbol side
RlQuote md::getVenueRoundLot(uint16_t s, Side sd) noexcept { return g_rlq[2u * s + static_cast<unsigned>(sd)]; }
static constexpr auto kNoCb = [](auto, uint64_t, auto) {};
// The host's book update has just written its own top of book: the venue record is
// in L1 when the tactic runs (the NBBO record, written by another process, is not).
static volatile PriceType g_sink;
static void book_touch(uint16_t s) { g_sink = g_vrec[s].q.bid; }
// The decode stage's prefetches for a symbol whose message is coming up. pf 1: the cancel tactic's own
// state (CancelTactic::prefetch); pf 2: and the NBBO record, which is the host's to warm. always_inline:
// a function that only prefetches is deleted by the compiler wherever it is not inlined.
template <class CTX>
[[gnu::always_inline]] static inline void decode_prefetch(const CTX& ct, uint16_t s, int pf) {
    ct.prefetch(s);
    if (pf > 1) __builtin_prefetch(&g_rec[s], 0, 3);
}
static const char* temp_name(bool cold, int pf) {
    static const char* k[2][3] = {{"hot", "hot+pf", "hot+pfq"}, {"cold", "cold+pf", "cold+pfq"}};
    return k[cold][pf];
}

// ---- our orders' queues: one level (QBlockList) per order -------------------------

struct Queues {
    QBlockPool pool;
    std::vector<QBlockList> lists;
    explicit Queues(std::size_t n) : pool(n + 1) { lists.reserve(n); }
    uint32_t push(uint64_t id, bool behind_a_round_lot = false) {
        lists.emplace_back();
        if (behind_a_round_lot) lists.back().push(pool, id + (1ull << 40), 200);
        return lists.back().push(pool, id, 100);
    }
};

static inline uint64_t tb() { _mm_lfence(); const uint64_t t = __rdtsc(); _mm_lfence(); return t; }
// A value the compiler cannot see through: keeps a timed call's constant
// arguments from folding its compares away (the host passes feed data).
template <class T> static inline T opaque(T x) { asm volatile("" : "+r"(x)); return x; }
static inline uint64_t te() { unsigned a; const uint64_t t = __rdtscp(&a); _mm_lfence(); return t; }
static uint64_t g_overhead = 0;

struct Q { double med, p99; };
static Q quant(std::vector<uint32_t>& v, uint64_t sub) {
    std::sort(v.begin(), v.end());
    auto at = [&](double f) { const double x = double(v[std::size_t(f * double(v.size() - 1))]) - double(sub);
                              return x < 0 ? 0.0 : x; };
    return {at(0.5), at(0.99)};
}

static std::vector<uint16_t> visit_order(unsigned NS, unsigned rounds, bool cold) {
    std::vector<uint16_t> perm(NS), out;
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(42);
    for (unsigned r = 0; r < rounds; ++r) {
        if (cold) std::shuffle(perm.begin(), perm.end(), rng);
        out.insert(out.end(), perm.begin(), perm.end());
    }
    return out;
}

// ---- part 1 -----------------------------------------------------------------

struct Ev { uint16_t sym; uint8_t kind; PriceType bid; Side rest; PriceType px; FeedTimeType ts; };
enum : uint8_t { kMoved, kSame, kExec, kKinds };
static const char* kName[kKinds] = {"nbbo moved", "nbbo same", "execute"};

// K = 0: untracked symbols unless pinned.
// qmode 0: no queue flags on any message (each order alone in its level).
//       1: removals (the Cancel and Execute messages) carry kQGone, as every real one does.
//       2: and each order rests behind a round-lot order, so (b) is armed for it.
static void steady(const char* label, unsigned NS, bool cold, unsigned K, bool full_path,
                   bool pinned = false, int pf = 0, int qmode = 0) {
    CancelParams p;
    p.aggr_prints_n = 2'000'000'000;               // (d) counts but never fires
    if (full_path) {                               // every order clears the cheap floors:
        p.min_ticks = 0; p.hist_spread_mult = 0.0; //   cheap floor = spread
        p.place_spread_mult = -1.0;                //   depth > move_min always
        p.gap_mult = 1e9; p.vol_horizon_cap_ns = 1'000'000'000'000'000ull;   // reach huge: no fire
    }
    Queues qs(std::size_t(NS) * 2 * K + 1);
    CT ct(NS, std::size_t(NS) * 2 * (K ? K : 1), qs.pool, p);
    g_rec.assign(NS, NbboRec{}); g_vrec.assign(NS, NbboRec{});
    uint64_t fired = 0;
    auto cb = [&](auto, uint64_t, auto) { ++fired; };
    FeedTimeType t = 1'000'000'000;
    const PriceType B = 1'000'000, A = 1'000'100;
    for (unsigned s = 0; s < NS; ++s) {
        const uint16_t y = uint16_t(s);
        ct.set_symbol(y, 100, 100);
        ct.set_tracking(y, true);                  // warm the estimators
        set_nbbo(y, B, A); ct.on_book_update(y, BookEvent::Add, Side::Buy, B, t, 0, cb);
        for (int k = 0; k < 4; ++k) { t += 1'000'000; ct.on_book_update(y, BookEvent::Execute, Side::Buy, B, t, 0, cb); }
        for (int k = 0; k < 8; ++k) {
            t += 1'000'000; const PriceType d = (k & 1) ? 100 : -100;
            set_nbbo(y, B + d, A + d); ct.on_book_update(y, BookEvent::Add, Side::Buy, B, t, 0, cb);
        }
        set_nbbo(y, B, A); ct.on_book_update(y, BookEvent::Add, Side::Buy, B, t + 1, 0, cb);
        for (unsigned k = 0; k < K; ++k) {         // K per side, 1..K ticks behind the touch
            const uint64_t ib = 1 + 2 * (uint64_t(s) * K + k), is = ib + 1;
            ct.on_place(ib, y, Side::Buy,  B - 100 * PriceType(k + 1), t + 1, qs.push(ib, qmode == 2), kNoCb);
            ct.on_place(is, y, Side::Sell, A + 100 * PriceType(k + 1), t + 1, qs.push(is, qmode == 2), kNoCb);
        }
        ct.set_tracking(y, pinned);
    }
    constexpr unsigned kEvents = 400'000;
    const auto order = visit_order(NS, (kEvents + NS - 1) / NS, cold);
    std::vector<Ev> evs; evs.reserve(kEvents);
    std::vector<uint32_t> visit(NS, 0);
    t += 10'000'000;
    for (unsigned i = 0; i < kEvents; ++i) {
        const uint16_t s = order[i];
        t += 50'000;
        const unsigned v = visit[s]++;
        switch (v % 4) {
        case 0:  evs.push_back({s, kMoved, (v / 4) & 1 ? B : B - 100, Side::Buy, 0, t}); break;   // bid toggles
        case 2:  { const bool bid_hit = (v / 4) & 1;                                              // a bid hit or
                   evs.push_back({s, kExec, 0, bid_hit ? Side::Buy : Side::Sell, bid_hit ? B - 100 : A, t}); break; }  // an offer lifted
        default: evs.push_back({s, kSame, 0, Side::Buy, 0, t}); break;
        }
    }
    const uint8_t gone = opaque(uint8_t(qmode ? kQGone : 0));
    auto apply = [&](const Ev& e) {
        switch (e.kind) {
        case kMoved: set_nbbo(e.sym, e.bid, A); ct.on_book_update(e.sym, BookEvent::Add, Side::Buy, e.bid, e.ts, 0, cb); break;
        case kExec:  ct.on_book_update(e.sym, BookEvent::Execute, e.rest, e.px, e.ts, gone, cb); break;
        default:     ct.on_book_update(e.sym, BookEvent::Cancel, Side::Sell, A, e.ts, gone, cb); break;
        }
    };
    const unsigned half = kEvents / 2;
    for (unsigned i = 0; i < half; ++i) apply(evs[i]);
    const uint64_t fired0 = fired;
    std::vector<uint32_t> samp[kKinds];
    for (unsigned i = half; i < kEvents; ++i) {
        book_touch(evs[i].sym);
        const uint64_t a = tb();
        if (pf && i + 2 < kEvents) decode_prefetch(ct, evs[i + 2].sym, pf);   // two updates ahead
        apply(evs[i]);
        const uint64_t b = te();
        samp[evs[i].kind].push_back(uint32_t(b - a));
    }
    char rowk[24];
    if (K) std::snprintf(rowk, sizeof rowk, "K=%u", K);
    else   std::snprintf(rowk, sizeof rowk, "%s", pinned ? "pinned" : "untracked");
    static const char* kQ[3] = {"", " +gone", " +armed"};
    std::printf("%-4s %-9s %-8s %-9s%-7s", label, full_path ? "full-path" : "typical", temp_name(cold, pf), rowk, kQ[qmode]);
    for (unsigned k = 0; k < kKinds; ++k) { const Q q = quant(samp[k], g_overhead); std::printf(" | %s %4.0f/%4.0f", kName[k], q.med, q.p99); }
    std::printf("%s\n", fired != fired0 ? "  ** FIRED **" : "");
}

// ---- part 2 -----------------------------------------------------------------

static uint64_t g_cb_tsc = 0;

static long thp_kb() {                                   // AnonHugePages of this process, kB
    long kb = -1;
    if (std::FILE* f = std::fopen("/proc/self/smaps_rollup", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f))
            if (std::strncmp(line, "AnonHugePages:", 14) == 0) kb = std::atol(line + 14);
        std::fclose(f);
    }
    return kb;
}

// armed: our order rests behind a round-lot order ((b) armed for it) and the
// removal messages — every Execute, Cancel and Delete here — carry kQGone, as
// the book's remove() reports it: the default configuration's usual state.
static void decision(const char* label, unsigned NS, bool cold, int which, int pf = 0, bool armed = false) {
    static const char* kWhich[2] = {"(d) on execute", "(c) on update "};
    constexpr int kTrain = 15;                    // non-firing updates of the same kind per trigger
    CancelParams p;
    p.aggr_prints_n = kTrain + 1;
    p.gap_mult = 1.0;
    const unsigned rounds = std::max(4u, 160'000u / NS);
    QBlockPool pool(std::size_t(NS) + 1);
    std::vector<QBlockList> lists(NS);            // one level per symbol: each round's order
    CT ct(NS, std::size_t(NS) * 2, pool, p);
    g_rec.assign(NS, NbboRec{}); g_vrec.assign(NS, NbboRec{});
    uint64_t fired = 0;
    auto cb = [&](auto h, uint64_t, auto) { _mm_lfence(); g_cb_tsc = __rdtsc(); ++fired; ct.on_done(h); };
    auto nocb = [](auto, uint64_t, auto) {};
    const uint8_t gone = opaque(uint8_t(armed ? kQGone : 0));
    std::vector<uint32_t> qm(NS, QBlockList::kBadHandle);            // armed: the round lot ahead of ours
    const PriceType B = 1'000'000, A = 1'000'100, AW = A + 1'000;   // AW: offer gapped 10 ticks
    FeedTimeType t = 1'000'000'000;
    for (unsigned s = 0; s < NS; ++s) {
        const uint16_t y = uint16_t(s);
        ct.set_symbol(y, 100, 100);
        ct.set_tracking(y, true);
        set_nbbo(y, B, A); ct.on_book_update(y, BookEvent::Add, Side::Buy, B, t, 0, nocb);
        for (int k = 0; k < 4; ++k) { t += 1'000'000; ct.on_book_update(y, BookEvent::Execute, Side::Buy, B, t, 0, nocb); }
        ct.set_tracking(y, false);
    }
    const long thp = thp_kb();
    const auto order = visit_order(NS, rounds, cold);
    std::vector<uint32_t> ttc, total;
    std::vector<uint32_t> qh(NS, QBlockList::kBadHandle);
    ttc.reserve(std::size_t(rounds) * NS); total.reserve(std::size_t(rounds) * NS);
    uint64_t next_id = 1;
    for (unsigned r = 0; r < rounds; ++r) {
        t += 100'000'000;                                          // setup: all symbols, in order
        for (unsigned s = 0; s < NS; ++s) {
            const uint16_t y = uint16_t(s);
            if (qh[s] != QBlockList::kBadHandle) lists[s].remove(pool, qh[s]);   // last round's order left
            if (qm[s] != QBlockList::kBadHandle) lists[s].remove(pool, qm[s]);
            if (armed) qm[s] = lists[s].push(pool, next_id + (1ull << 40), 200);
            qh[s] = lists[s].push(pool, next_id, 100);
            set_nbbo(y, B, A);
            if (which == 0) {                                      // kTrain counted executions, no fire
                ct.on_place(next_id, y, Side::Buy, B - 300, t + 1, qh[s], kNoCb);
                for (int k = 0; k < kTrain; ++k)
                    ct.on_book_update(y, BookEvent::Execute, Side::Buy, B, t + 1'000 * (k + 1), gone, nocb);
            } else {                                               // kTrain evaluations rejected at
                ct.on_place(next_id, y, Side::Buy, B - 200, t + 1, qh[s], kNoCb);   // the cheap floors
                for (int k = 0; k < kTrain; ++k)
                    ct.on_book_update(y, BookEvent::Cancel, Side::Sell, A, t + 1'000 * (k + 1), gone, nocb);
                set_nbbo(y, B, AW);                                // then the far touch gaps away
                ct.on_book_update(y, BookEvent::Delete, Side::Sell, A, t + 20'000, gone, nocb);
            }
            ++next_id;
        }
        const FeedTimeType tt = t + 500'000;                       // gate open (gap seen at t+20us)
        for (unsigned i = 0; i < NS; ++i) {                        // triggers, hot or random order
            const uint16_t y = order[std::size_t(r) * NS + i];
            const uint64_t f0 = fired;
            book_touch(y);
            const uint64_t a = tb();
            if (pf && i + 2 < NS) decode_prefetch(ct, order[std::size_t(r) * NS + i + 2], pf);
            if (which == 0) ct.on_book_update(y, BookEvent::Execute, Side::Buy, B, tt, gone, cb);
            else if (armed) ct.on_book_update(y, BookEvent::Cancel, Side::Buy, B, tt, gone, cb);   // our side
            else            ct.on_book_update(y, BookEvent::Cancel, Side::Sell, AW, tt, 0, cb);     // NBBO unchanged
            const uint64_t b = te();
            if (fired == f0 + 1) { ttc.push_back(uint32_t(g_cb_tsc - a)); total.push_back(uint32_t(b - a)); }
        }
    }
    const Q q1 = quant(ttc, g_overhead / 2), q2 = quant(total, g_overhead);
    std::printf("%-4s %s %-8s%s | to callback %4.0f/%4.0f | whole call %4.0f/%4.0f | fired %zu/%zu | THP %ld MB\n",
                label, kWhich[which], temp_name(cold, pf), armed ? " armed" : "", q1.med, q1.p99, q2.med, q2.p99,
                ttc.size(), std::size_t(rounds) * NS, thp / 1024);
}


// ---- part 12: (a) and (b) decisions, call entry -> callback entry -----------------
// (b) at the front: our order rests behind one round-lot order; after 15
// removals at another price (no fire), that order is deleted — the book's
// remove() runs first, untimed, as it does in the host — and the update
// carrying its flags finds ours at the front with 1 of 1 cancelled.
// (b) behind an odd lot: the same with an odd lot ahead of both, which stays.
// (b) on a cut: two round lots ahead, X = 40%; one is deleted in the setup
// (1 of 2, the other still holds (b) back), and the timed update is the
// partial execution that leaves that one an odd lot.
// (b), three at the price: one round lot, then three orders of ours; its
// removal decides all three (to callback: the first of them).
// (a): three orders of 90 ahead of our 100 (N = 3, K = 3: 270 < 300); after
// 15 such updates our order is cut to 80 (270 >= 240) and the update
// carrying kQTracked fires.
static void decision_q(const char* label, unsigned NS, bool cold, int which) {
    static const char* kWhich[5] = {"(b) at the front      ", "(a) ours reduced      ", "(b) behind an odd lot ",
                                    "(b) on a cut          ", "(b) three at the price"};
    constexpr int kTrain = 15;
    CancelParams p;
    p.aggr_prints_n = 2'000'000'000;
    p.queue_n = which == 1 ? 3 : 0; p.queue_k = 3.0; p.abandon_pct = which == 1 ? -1.0 : which == 3 ? 40.0 : 50.0;
    const unsigned rounds = std::max(4u, 160'000u / NS);
    const unsigned mine = which == 4 ? 3 : 1;                     // orders of ours per symbol
    QBlockPool pool(std::size_t(NS) + 1);
    std::vector<QBlockList> lists(NS);
    CT ct(NS, std::size_t(NS) * 4, pool, p);
    g_rec.assign(NS, NbboRec{}); g_vrec.assign(NS, NbboRec{});
    for (auto& st : g_st) { st.tac = TacticShared{}; st.round_lot = 100; }
    uint64_t fired = 0, mark = 0;
    auto cb = [&](auto h, uint64_t, auto) {
        _mm_lfence(); const uint64_t now = __rdtsc();
        if (fired == mark) g_cb_tsc = now;                        // the first of this update's decisions
        ++fired; ct.on_done(h);
    };
    auto nocb = [](auto, uint64_t, auto) {};
    const PriceType B = 1'000'000, A = 1'000'100, P = B - 300;
    FeedTimeType t = 1'000'000'000;
    for (unsigned s = 0; s < NS; ++s) {
        const uint16_t y = uint16_t(s);
        ct.set_symbol(y, 100, 100);
        set_nbbo(y, B, A);
    }
    const auto order = visit_order(NS, rounds, cold);
    std::vector<uint32_t> ttc, total;
    struct Lv { uint32_t q[6]; unsigned n = 0; uint32_t hit = 0; };  // hit: the order the timed message is about
    std::vector<Lv> lv(NS);
    uint64_t next_id = 1;
    for (unsigned r = 0; r < rounds; ++r) {
        t += 100'000'000;
        for (unsigned s = 0; s < NS; ++s) {                       // setup: all symbols, in order
            const uint16_t y = uint16_t(s);
            Lv& L = lv[s];
            for (unsigned k = 0; k < L.n; ++k) lists[s].remove(pool, L.q[k]);
            L.n = 0;
            uint32_t first = 0;                                   // (b) on a cut: deleted in the setup
            if (which == 1) {
                for (int k = 0; k < 3; ++k) L.q[L.n++] = lists[s].push(pool, next_id + (uint64_t(k + 1) << 40), 90);
            } else {
                if (which == 2) L.q[L.n++] = lists[s].push(pool, next_id + (2ull << 40), 50);
                if (which == 3) first = lists[s].push(pool, next_id + (2ull << 40), 200);
                L.hit = L.q[L.n++] = lists[s].push(pool, next_id + (1ull << 40), which == 3 ? 300 : 200);
            }
            for (unsigned k = 0; k < mine; ++k) {
                const uint32_t qo = lists[s].push(pool, next_id + (uint64_t(k) << 44), 100);
                L.q[L.n++] = qo;
                if (which == 1) L.hit = qo;
                ct.on_place(next_id + (uint64_t(k) << 44), y, Side::Buy, P, t + 1, qo, kNoCb);
            }
            for (int k = 0; k < kTrain; ++k)
                ct.on_book_update(y, BookEvent::Delete, Side::Buy, P - 100, t + 1'000 * (k + 1), kQGone, nocb);
            if (which == 3)                                       // 1 of 2 cancelled: the ratio is met
                ct.on_book_update(y, BookEvent::Delete, Side::Buy, P, t + 20'000, lists[s].remove(pool, first), nocb);
            ++next_id;
        }
        const FeedTimeType tt = t + 500'000;
        for (unsigned i = 0; i < NS; ++i) {                        // triggers, hot or random order
            const uint16_t y = order[std::size_t(r) * NS + i];
            Lv& L = lv[y];
            const uint64_t f0 = fired;
            mark = fired;
            uint8_t f;                                             // the book's own work, untimed
            if (which == 1)      f = lists[y].reduce(pool, L.hit, 20);
            else if (which == 3) f = lists[y].reduce(pool, L.hit, 250);
            else {
                f = lists[y].remove(pool, L.hit);
                for (unsigned k = 0; k < L.n; ++k) if (L.q[k] == L.hit) { L.q[k] = L.q[--L.n]; break; }
            }
            book_touch(y);
            const PriceType px = opaque(P);
            f = opaque(f);
            const BookEvent ev = which == 1 ? BookEvent::Cancel : which == 3 ? BookEvent::Execute : BookEvent::Delete;
            const uint64_t a = tb();
            ct.on_book_update(y, ev, Side::Buy, px, tt, f, cb);
            const uint64_t b = te();
            if (fired == f0 + mine) { ttc.push_back(uint32_t(g_cb_tsc - a)); total.push_back(uint32_t(b - a)); }
        }
    }
    const Q q1 = quant(ttc, g_overhead / 2), q2 = quant(total, g_overhead);
    std::printf("%-4s %s %-4s | to callback %4.0f/%4.0f | whole call %4.0f/%4.0f | fired %zu/%zu\n",
                label, kWhich[which], cold ? "cold" : "hot", q1.med, q1.p99, q2.med, q2.p99,
                ttc.size(), std::size_t(rounds) * NS);
}

// ---- part 9: join tactic ----------------------------------------------------

// The host's two call sites, one per entry point, each in a function of its
// own as in a feed handler: what the handler pays, call included.
static uint64_t g_sig = 0;
[[gnu::noinline]] static void host_add(JoinTactic& jt, uint16_t y, Side sd, PriceType px, uint32_t sz,
                                       PriceType prev, FeedTimeType t) {
    jt.on_add(y, sd, px, sz, prev, t, [](uint16_t, Side, const JoinSide&) { ++g_sig; });
}
[[gnu::noinline]] static void host_exec(JoinTactic& jt, uint16_t y, Side rest, PriceType px, uint32_t sz) {
    jt.on_execute(y, rest, px, sz);
}

static void join_paths(const char* label, unsigned NS, bool cold, bool pf = false) {
    for (unsigned s = 0; s < NS; ++s) { g_st[s].tac = TacticShared{}; g_st[s].round_lot = 100; }
    const PriceType B = 1'000'000, A = 1'000'300;
    for (unsigned s = 0; s < NS; ++s) { g_rlq[2 * s] = {B, 300}; g_rlq[2 * s + 1] = {A, 300}; }
    JoinTactic jt(NS), none(NS);
    for (unsigned s = 0; s < NS; ++s) jt.arm(uint16_t(s), true);
    uint64_t sig = 0;
    auto cb = [&](uint16_t, Side, const JoinSide&) { ++sig; };
    constexpr unsigned kN = 200'000;
    const auto order = visit_order(NS, (kN + NS - 1) / NS, cold);
    enum { kUnarmed, kBehind, kOdd, kLive, kSignal, kExecNone, kExecOne, kExecSig, kExecKill, kRows };
    static const char* name[kRows] = {"unarmed", "add behind", "add odd lot", "add, sig live", "add SIGNAL",
                                      "exec no sig", "exec, 1 sig", "exec, 2 sigs", "exec reads kill"};
    std::vector<uint32_t> v[kRows];
    FeedTimeType t = 1'000'000'000;
    auto end_all = [&] {
        for (unsigned s = 0; s < NS; ++s) {
            JoinTactic::kill(uint16_t(s), Side::Buy); JoinTactic::kill(uint16_t(s), Side::Sell); jt.poll(uint16_t(s));
        }
    };
    auto signal_all = [&] {                                         // a live signal on both sides
        for (unsigned s = 0; s < NS; ++s) {
            jt.on_add(uint16_t(s), Side::Buy,  B + 100, 500, B, t, cb);
            jt.on_add(uint16_t(s), Side::Sell, A - 100, 500, A, t, cb);
        }
    };
    for (int row = 0; row < kRows; ++row) {
        const bool live = row == kLive || row == kSignal || row == kExecSig || row == kExecKill;
        end_all();
        if (live) signal_all();
        if (row == kExecOne)                                        // the bid side's signal only
            for (unsigned s = 0; s < NS; ++s) jt.on_add(uint16_t(s), Side::Buy, B + 100, 500, B, t, cb);
        for (unsigned i = 0; i < kN; ++i) {
            const uint16_t y = order[i];
            t += 1'000;
            if (row == kSignal || row == kExecKill) JoinTactic::kill(y, Side::Buy);   // the strategy's flag
            const PriceType up = opaque(B + 100), dn = opaque(B - 100), bb = opaque(B), at = opaque(B);
            const uint32_t big = opaque(500u), odd = opaque(50u), q = opaque(100u);
            const uint64_t a = tb();
            if (pf && i + 2 < kN) jt.prefetch(order[i + 2]);        // decode-stage prefetch, 2 messages ahead
            switch (row) {
            case kUnarmed:  host_add(none, y, Side::Buy, up, big, bb, t); break;
            case kBehind:   host_add(jt, y, Side::Buy, dn, big, bb, t); break;
            case kOdd:      host_add(jt, y, Side::Buy, up, odd, bb, t); break;
            case kLive:
            case kSignal:   host_add(jt, y, Side::Buy, up, big, bb, t); break;
            case kExecNone: host_exec(jt, y, Side::Buy, at, q); break;
            default:        host_exec(jt, y, Side::Buy, up, q); break;
            }
            const uint64_t b = te();
            if (i >= kN / 2) v[row].push_back(uint32_t(b - a));
            if (row == kExecKill) jt.on_add(y, Side::Buy, B + 100, 500, B, t, cb);   // signals again
            if (cold) {
                char* tac = reinterpret_cast<char*>(&g_st[y].tac);
                for (std::size_t o = 0; o < sizeof(TacticShared); o += 64) _cldemote(tac + o);
                _cldemote(&g_rlq[2u * y]);
            }
        }
    }
    std::printf("%-5s join  %-7s NS=%-5u", label, cold ? (pf ? "cold+pf" : "cold") : (pf ? "hot+pf" : "hot"), NS);
    for (int r = 0; r < kRows; ++r) { const Q q = quant(v[r], g_overhead); std::printf(" | %s %3.0f/%3.0f", name[r], q.med, q.p99); }
    std::printf("\n");
    g_sink = PriceType(sig + g_sig);
}

// ---- part 10: cancel (a)/(b) paths -----------------------------------------

static void queue_paths(const char* label) {
    const PriceType B = 1'000'000, A = 1'000'100, P = B - 100;
    g_rec.assign(4, NbboRec{}); g_vrec.assign(4, NbboRec{});
    set_nbbo(1, B, A);
    // on_place: the walk over the queue ahead; then the same with a live join
    // signal whose tape is full (16 executions a side), (d) seeded from it
    for (uint32_t ahead : {1u, 8u, 15u, 40u}) {
        double med[3], p99[3];
        for (int on = 0; on < 3; ++on) {
            for (auto& st : g_st) { st.tac = TacticShared{}; st.round_lot = 100; }
            if (on == 2) {
                JoinTactic jt(4); jt.arm(1, true);
                jt.on_add(1, Side::Buy, B, 500, P, 1, [](uint16_t, Side, const JoinSide&) {});
                for (int k = 0; k < 40; ++k) jt.on_execute(1, k & 1 ? Side::Sell : Side::Buy, B + 100 * (k % 7), 100);
            }
            QBlockPool pool(64);
            QBlockList lvl;
            for (uint32_t k = 0; k < ahead; ++k) lvl.push(pool, 1000 + k, (k & 1) ? 50 : 200);
            CancelParams p;
            p.aggr_prints_n = 2'000'000'000;
            p.queue_n = 0; p.abandon_pct = -1.0;
            if (on) { p.queue_n = 1'000'000; p.abandon_pct = 50.0; }   // walk, never fire
            p.aggr_from_signal = on == 2;
            CT ct(4, 8, pool, p); ct.set_symbol(1, 100, 100);
            std::vector<uint32_t> v;
            FeedTimeType t = 1'000'000'000;
            for (unsigned i = 0; i < 100'000; ++i) {
                const uint32_t qh = lvl.push(pool, 7, 100);
                t += 1'000;
                const PriceType px = opaque(P);
                const uint64_t a = tb();
                const auto h = ct.on_place(7, 1, Side::Buy, px, t, qh, kNoCb);
                const uint64_t b = te();
                if (on == 2 && ct.order(h)->n_aggr_prints == 0) std::printf("  ** not seeded **\n");
                ct.on_done(h);
                lvl.remove(pool, qh);
                if (i >= 50'000) v.push_back(uint32_t(b - a));
            }
            const Q q = quant(v, g_overhead); med[on] = q.med; p99[on] = q.p99;
        }
        std::printf("%-5s (a)/(b) on_place, %2u ahead | off %4.0f/%4.0f | on %4.0f/%4.0f | on + join seed %4.0f/%4.0f\n",
                    label, ahead, med[0], p99[0], med[1], p99[1], med[2], p99[2]);
    }
    for (auto& st : g_st) { st.tac = TacticShared{}; st.round_lot = 100; }
    // removal messages for a symbol with one (b)-armed order, not at the front;
    // our order reduced: (a) re-checked (never fires)
    QBlockPool pool(64);
    QBlockList lvl;
    for (uint32_t k = 0; k < 4; ++k) lvl.push(pool, 1000 + k, 200);
    CancelParams p;
    p.aggr_prints_n = 2'000'000'000; p.abandon_pct = 50.0; p.queue_n = 1'000'000;
    CT ct(4, 8, pool, p); ct.set_symbol(1, 100, 100);
    FeedTimeType t = 1'000'000'000;
    ct.on_place(7, 1, Side::Buy, P, t, lvl.push(pool, 7, 100), kNoCb);
    lvl.push(pool, 2000, 300);                                       // behind us
    enum { kOurs, kOther, kNone, kReduced, kRows };
    static const char* name[kRows] = {"removal at our price", "removal elsewhere", "no queue flags",
                                      "ours reduced, (a) again"};
    std::vector<uint32_t> v[kRows];
    for (unsigned i = 0; i < 400'000; ++i) {
        const int row = int(i % kRows);
        t += 1'000;
        const PriceType ours = opaque(P), other = opaque(P - 100);
        const uint8_t fa = opaque(uint8_t(kQGone | kQAhead)), fg = opaque(uint8_t(kQGone)), f0 = opaque(uint8_t(0)),
                      ft = opaque(uint8_t(kQTracked));
        const uint64_t a = tb();
        if (row == kOurs)       ct.on_book_update(1, BookEvent::Delete, Side::Buy, ours, t, fa, kNoCb);
        else if (row == kOther) ct.on_book_update(1, BookEvent::Delete, Side::Buy, other, t, fg, kNoCb);
        else if (row == kNone)  ct.on_book_update(1, BookEvent::Delete, Side::Buy, other, t, f0, kNoCb);
        else                    ct.on_book_update(1, BookEvent::Cancel, Side::Buy, ours, t, ft, kNoCb);
        const uint64_t b = te();
        if (i >= 200'000) v[row].push_back(uint32_t(b - a));
    }
    std::printf("%-5s (a)/(b) one order, hot", label);
    for (int r = 0; r < kRows; ++r) { const Q q = quant(v[r], g_overhead); std::printf(" | %s %3.0f/%3.0f", name[r], q.med, q.p99); }
    std::printf("\n");
    // (b) once an order's ratio is met: every removal at its price, and every cut of
    // a flagged order there, then reads the sizes ahead of it. One round lot is left,
    // at the front of the queue — the far end of that read — behind it odd lots only.
    for (uint32_t ahead : {4u, 15u, 40u, 200u}) {
        QBlockPool pl(64);
        QBlockList lv;
        std::vector<uint32_t> hs;
        for (uint32_t k = 0; k < 3; ++k) hs.push_back(lv.push(pl, 900 + k, 200));        // will cancel
        lv.push(pl, 1000, 200);                                                          // stays
        for (uint32_t k = 1; k < ahead; ++k) lv.push(pl, 1000 + k, 50);
        CancelParams pp;
        pp.aggr_prints_n = 2'000'000'000; pp.abandon_pct = 50.0; pp.queue_n = 0;
        CT c2(4, 8, pl, pp); c2.set_symbol(1, 100, 100);
        FeedTimeType u = 1'000'000'000;
        const auto h = c2.on_place(7, 1, Side::Buy, P, u, lv.push(pl, 7, 100), kNoCb);
        for (uint32_t qh : hs) c2.on_book_update(1, BookEvent::Delete, Side::Buy, P, u += 1'000, lv.remove(pl, qh), kNoCb);
        if (c2.order(h)->ahead_cancelled != 3 || !c2.order(h)->active) std::printf("  ** setup **\n");
        std::vector<uint32_t> w[2];
        for (unsigned i = 0; i < 200'000; ++i) {
            u += 1'000;
            const PriceType ours = opaque(P);
            const uint8_t fg = opaque(uint8_t(kQGone)), fc = opaque(uint8_t(kQAhead));
            const uint64_t a = tb();
            if (i & 1) c2.on_book_update(1, BookEvent::Execute, Side::Buy, ours, u, fc, kNoCb);
            else       c2.on_book_update(1, BookEvent::Delete, Side::Buy, ours, u, fg, kNoCb);
            const uint64_t b = te();
            if (i >= 100'000) w[i & 1].push_back(uint32_t(b - a));
        }
        const Q qa = quant(w[0], g_overhead), qb = quant(w[1], g_overhead);
        std::printf("%-5s (b) ratio met, a round lot %3u ahead | removal at our price %3.0f/%3.0f | cut of a flagged order %3.0f/%3.0f%s\n",
                    label, ahead, qa.med, qa.p99, qb.med, qb.p99, c2.order(h)->active ? "" : "  ** FIRED **");
    }
}

// ---- part 15: on_place_late ---------------------------------------------------------
// An order registered at the exchange's ack, after its Add: on_place_late
// against on_place for the same order in the same book. Rows: alone at its
// price; behind four round lots; behind four round lots with one of ours
// behind it that counted it as a market round lot (the correction); at the
// front of a level whose join signal it raised (the kill). Hot; the cost
// does not depend on how late the call is.
static void late_paths(const char* label) {
    const PriceType B = 1'000'000, A = 1'000'100, P = B - 100;
    g_rec.assign(4, NbboRec{}); g_vrec.assign(4, NbboRec{});
    set_nbbo(1, B, A);
    static const char* rows[4] = {"alone at its price", "behind four round lots", "and one of ours behind it",
                                  "its own join signal"};
    for (int row = 0; row < 4; ++row) {
        double med[2], p99[2];
        for (int late = 0; late < 2; ++late) {
            for (auto& st : g_st) { st.tac = TacticShared{}; st.round_lot = 100; }
            QBlockPool pool(64);
            QBlockList lvl;
            if (row == 1 || row == 2) for (uint32_t k = 0; k < 4; ++k) lvl.push(pool, 1000 + k, 200);
            CancelParams p;
            p.aggr_prints_n = 2'000'000'000; p.queue_n = 1'000'000; p.abandon_pct = 50.0;
            CT ct(4, 8, pool, p); ct.set_symbol(1, 100, 100); ct.set_tracking(1, true);
            std::vector<uint32_t> v;
            FeedTimeType t = 1'000'000'000;
            for (unsigned i = 0; i < 40'000; ++i) {
                const uint32_t qh = lvl.push(pool, 7, 200);
                ct.on_book_update(1, BookEvent::Add, Side::Buy, P, t += 1'000, uint8_t{0}, kNoCb);
                const FeedTimeType t_add = t;
                CT::Handle hb = CT::kInvalid;
                uint32_t qb = 0;
                if (row == 2) {                                   // one of ours behind it, counting it
                    qb = lvl.push(pool, 8, 100);
                    hb = ct.on_place(8, 1, Side::Buy, P, t += 1'000, qb, kNoCb);
                    ct.on_book_update(1, BookEvent::Add, Side::Buy, P, t, uint8_t{0}, kNoCb);
                }
                if (row == 3) {                                   // the signal it raised
                    JoinSide& js = g_st[1].tac.side[0];
                    js.px = P; js.t = t_add; js.kill = 0;
                }
                ct.on_book_update(1, BookEvent::Add, Side::Sell, A + 200, t += 1'000, uint8_t{0}, kNoCb);
                const PriceType px = opaque(P);
                const uint64_t a = tb();
                const auto h = late ? ct.on_place_late(7, 1, Side::Buy, px, t_add, qh, kNoCb)
                                    : ct.on_place(7, 1, Side::Buy, px, t_add, qh, kNoCb);
                const uint64_t b = te();
                if (i == 0 && h == CT::kInvalid) std::printf("  ** setup **\n");
                ct.on_done(h);
                ct.on_book_update(1, BookEvent::Delete, Side::Buy, P, t += 1'000, lvl.remove(pool, qh), kNoCb);
                if (row == 2) {
                    ct.on_done(hb);
                    ct.on_book_update(1, BookEvent::Delete, Side::Buy, P, t += 1'000, lvl.remove(pool, qb), kNoCb);
                }
                if (i >= 20'000) v.push_back(uint32_t(b - a));
            }
            const Q q = quant(v, g_overhead); med[late] = q.med; p99[late] = q.p99;
        }
        std::printf("%-5s late  %-26s | on_place %4.0f/%4.0f | on_place_late %4.0f/%4.0f\n", label, rows[row], med[0],
                    p99[0], med[1], p99[1]);
    }
}

// ---- part 14: one handler, one call site per entry point --------------------------
// What a feed handler pays per message when it calls both tactics from one
// function — registering our own orders there too — as a host does (parts
// 1-2 time on_book_update inlined at several
// sites of the bench's own loop, where the compiler keeps the passes out of
// line whatever the header says). Rows: a symbol with no orders of ours, a
// pinned one, one order alone in its level, one order behind a round lot ((b)
// armed) with the messages at another price, and the same with them at our
// price. Removals carry kQGone.
struct HostMsg { uint16_t sym; BookEvent ev; Side sd; PriceType px; FeedTimeType ts; uint8_t qf; uint32_t sz; PriceType prev;
                 uint64_t ours; uint32_t qh; };          // ours: our order's id when the Add is our own, else 0
static uint64_t g_host_fired = 0;
[[gnu::noinline]] static void host_handle(CT& ct, JoinTactic& jt, const HostMsg& m) {
    const auto cb = [](auto, uint64_t, auto) { ++g_host_fired; };
    if (m.ev == BookEvent::Add) {
        if (m.ours) ct.on_place(m.ours, m.sym, m.sd, m.px, m.ts, m.qh, cb);          // our own order: register it
        else jt.on_add(m.sym, m.sd, m.px, m.sz, m.prev, m.ts, [](uint16_t, Side, const JoinSide&) { ++g_host_fired; });
    } else if (m.ev == BookEvent::Execute) {
        jt.on_execute(m.sym, m.sd, m.px, m.sz);
    }
    ct.on_book_update(m.sym, m.ev, m.sd, m.px, m.ts, m.qf, cb);
}

static void host_paths(const char* label) {
    constexpr unsigned NS = 32;                        // 8 symbols per row, round-robin (hot)
    const PriceType B = 1'000'000, A = 1'000'100;
    g_rec.assign(NS, NbboRec{}); g_vrec.assign(NS, NbboRec{});
    for (auto& st : g_st) { st.tac = TacticShared{}; st.round_lot = 100; }
    CancelParams p;
    p.aggr_prints_n = 2'000'000'000;
    Queues qs(64);
    CT ct(NS, 64, qs.pool, p);
    JoinTactic jt(NS);
    FeedTimeType t = 1'000'000'000;
    for (unsigned s = 0; s < NS; ++s) {
        const uint16_t y = uint16_t(s);
        ct.set_symbol(y, 100, 100); set_nbbo(y, B, A);
        if (s >= 8 && s < 16) ct.set_tracking(y, true);
        if (s >= 16) ct.on_place(1000 + s, y, Side::Buy, B - 300, t, qs.push(1000 + s, s >= 24), kNoCb);
    }
    static const char* rows[5] = {"no orders of ours", "pinned, no orders", "one order", "one order, (b) armed",
                                  "(b) armed, at our price"};
    static const BookEvent kinds[3] = {BookEvent::Add, BookEvent::Delete, BookEvent::Execute};
    static const char* names[3] = {"Add", "Delete", "Execute"};
    for (unsigned row = 0; row < 5; ++row) {
        std::printf("%-5s host  %-23s", label, rows[row]);
        for (int k = 0; k < 3; ++k) {
            std::vector<uint32_t> v;
            for (unsigned i = 0; i < 400'000; ++i) {
                const PriceType px = row == 4 ? B - 300 : B - 100 * PriceType(i & 1);
                HostMsg m{uint16_t(std::min(row, 3u) * 8 + (i & 7)), kinds[k], Side::Buy, px, t += 1'000,
                          uint8_t(k ? kQGone : 0), 100, B, 0, 0};
                asm volatile("" : : "r"(&m) : "memory");
                const uint64_t a = tb();
                host_handle(ct, jt, m);
                const uint64_t b = te();
                if (i >= 200'000) v.push_back(uint32_t(b - a));
            }
            const Q q = quant(v, g_overhead);
            std::printf(" | %s %3.0f/%3.0f", names[k], q.med, q.p99);
        }
        std::printf("%s\n", g_host_fired ? "  ** FIRED **" : "");
    }
}

// ---- part 8 -----------------------------------------------------------------

static void roundlot_hook(const char* label) {
    MarketDataSystem m(64, 1u << 16, 4096, 1u << 12);
    OrderRef ref = 1; UniqueUpdateID uu = 1;
    const FeedTimeType t = 1'000'000'000;
    for (uint16_t s : {uint16_t(1), uint16_t(2)}) {         // 10 levels a side
        m.configure_symbol(s, 100, 100);
        for (int i = 0; i < 10; ++i) {                      // sym 2: 40-share odd lots on the top two
            const SizeType q = s == 1 ? 300 : (i < 2 ? 40 : 200);
            m.on_add(ref++, s, Side::Buy,  1000000 - 100 * i, q, t, uu++);
            m.on_add(ref++, s, Side::Sell, 1000100 + 100 * i, q, t, uu++);
        }
    }
    auto hook = [&m](uint16_t s) {                          // the body suggested in mdsys_2.hpp
        const auto b = m.roundlot_best(s, Side::Buy), a = m.roundlot_best(s, Side::Sell);
        return Nbbo{b ? b->price : 0, a ? a->price : 0};
    };
    for (uint16_t s : {uint16_t(1), uint16_t(2)}) {
        std::vector<uint32_t> v(200'000);
        for (auto& x : v) {
            const uint64_t a = tb();
            const Nbbo r = hook(s);
            g_sink = r.bid + r.ask;
            const uint64_t b = te();
            x = uint32_t(b - a);
        }
        const Nbbo r = hook(s);
        const Q q = quant(v, g_overhead);
        std::printf("%-5s round-lot hook body, %s (%lld/%lld) | %4.0f/%4.0f\n", label,
                    s == 1 ? "a round lot at each best level" : "odd lots on the top two levels",
                    (long long)r.bid, (long long)r.ask, q.med, q.p99);
    }
}

int main(int argc, char** argv) {
    const char* label = argc > 1 ? argv[1] : "?";
    const int part = argc > 2 ? std::atoi(argv[2]) : 0;
    std::vector<uint32_t> ov(200'000);
    for (auto& x : ov) { const uint64_t a = tb(); const uint64_t b = te(); x = uint32_t(b - a); }
    std::sort(ov.begin(), ov.end());
    g_overhead = ov[ov.size() / 2];
    if (part == 0 || part == 1)
        for (bool full : {false, true})
            for (bool cold : {false, true}) {
                if (!full) { steady(label, cold ? 8192 : 8, cold, 0, false, false);
                             steady(label, cold ? 8192 : 8, cold, 0, false, true); }
                for (unsigned K : {1u, 4u, 8u}) steady(label, cold ? 8192 : 8, cold, K, full);
            }
    if (part == 0 || part == 2)
        for (int which : {0, 1}) {
            for (bool cold : {false, true}) decision(label, cold ? 8192 : 8, cold, which);
            for (int pf : {1, 2}) decision(label, 8192, true, which, pf);
        }
    if (part == 7) for (int which : {0, 1}) decision(label, 8, false, which);   // hot decisions
    if (part == 13)                                    // (c), (d) with (b) armed and real removal flags
        for (int which : {0, 1})
            for (bool cold : {false, true}) decision(label, cold ? 8192 : 8, cold, which, 0, true);
    if (part == 5) decision(label, 8192, true, 0);   // (d) on execute, cold only
    if (part == 6) decision(label, 8192, true, 1);   // (c) on update, cold only
    if (part == 8) roundlot_hook(label);
    if (part == 9) {
        join_paths(label, 8, false); join_paths(label, 8, false, true);
        join_paths(label, 8192, true); join_paths(label, 8192, true, true);
    }
    if (part == 10) queue_paths(label);
    if (part == 14) host_paths(label);
    if (part == 12)
        for (int which : {0, 2, 3, 4, 1})
            for (bool cold : {false, true}) decision_q(label, cold ? 8192 : 8, cold, which);
    if (part == 15) late_paths(label);
    if (part == 11)                                    // the default configuration's steady state
        for (bool cold : {false, true})
            for (unsigned K : {1u, 4u, 8u})
                for (int qm : {0, 1, 2}) steady(label, cold ? 8192 : 8, cold, K, false, false, false, qm);
    if (part == 3) {                                   // part 1 with and without the decode-stage prefetch
        for (int pf : {0, 1}) steady(label, 8, false, 0, false, false, pf);
        for (unsigned K : {1u, 8u}) {
            for (int pf : {0, 1}) steady(label, 8, false, K, false, false, pf);
            for (int pf : {0, 1, 2}) steady(label, 8192, true, K, false, false, pf);
        }
    }
    if (std::FILE* f = std::fopen("/proc/self/smaps_rollup", "r")) {     // confirm huge-page backing
        char line[256];
        while (std::fgets(line, sizeof line, f))
            if (std::strncmp(line, "AnonHugePages:", 14) == 0) std::fprintf(stderr, "[%s] %s", label, line);
        std::fclose(f);
    }
    std::fprintf(stderr, "timer overhead %llu TSC subtracted; cells = median/p99 TSC ticks\n",
                 (unsigned long long)g_overhead);
}
