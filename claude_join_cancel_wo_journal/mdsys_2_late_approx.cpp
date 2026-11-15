// mdsys_2_late_approx.cpp — how close an order registered at the exchange's ack
// (CancelTactic::on_place_late) comes to the same order registered at its Add
// (on_place), for mdsys_2.hpp [4]. A measurement, not a test: cancel_tactic.pdf,
// Appendix F.3, quotes its output.
//
// One random stream drives two tactics over books kept identical: A is told of
// each order of ours at its Add, B at its ack, 1 to <max lag> of the symbol's
// book updates later (the Add's own included). Decisions are never answered, so
// a decided order stays in the book, inactive in its tactic, and the books never
// diverge; each order's first decision in each tactic is compared. An order that
// leaves the book before its ack is never registered in B and is counted apart.
// The streams are much busier at our price than a real book: read the output as
// a stress, not an estimate. Join signals are killed only when no ack is
// pending, as the host contract asks (kill a signal after the acks of the orders
// placed on it).
//
// Build: g++ -std=c++23 -O2 -march=graniterapids -I. mdsys_2_late_approx.cpp -o late_approx
// Run:   ./late_approx [streams=1000] [max lag=60] [several of ours a price=0]
//                      [default parameters=1] [(d) not seeded from the join signal=0]
#include "mdsys_2.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <unordered_map>
#include <vector>

using namespace md;

static Nbbo g_nbbo[8];
Nbbo md::getNBBO(uint16_t s) noexcept { return g_nbbo[s]; }
static std::vector<SymbolState> g_st(8);
SymbolState& md::getSymbolState(uint16_t s) noexcept { return g_st[s]; }
static RlQuote g_rl[8][2];
RlQuote md::getVenueRoundLot(uint16_t s, Side sd) noexcept { return g_rl[s][static_cast<unsigned>(sd)]; }
Nbbo md::getVenueRoundLotBBO(uint16_t) noexcept { return {0, 0}; }

static constexpr PriceType T = 100;
using CT = CancelTactic;
using H = CT::Handle;
using D = CT::Decision;

struct Got { uint64_t id; D d; };
struct World {
    uint16_t sym;
    QBlockPool pool;
    CT ct;
    JoinTactic jt;
    std::map<std::pair<int, PriceType>, QBlockList> q;
    std::unordered_map<uint64_t, uint32_t> qh;
    std::vector<Got> got;
    World(uint16_t s, const CancelParams& p) : sym(s), pool(4096), ct(4, 64, pool, p), jt(4) {}
    auto cb() { return [this](H, uint64_t id, D d) { got.push_back({id, d}); }; }
};
struct Ord {
    int side = 0; PriceType px = 0; uint32_t sz = 0;
    bool ours = false;
    int lag = -1, lag0 = 0;                  // book updates until B learns of it; -1: B knows (or not ours)
    FeedTimeType t_add = 0;
    long ev_add = 0;
    H h[2] = {0, 0};
    long first_ev[2] = {-1, -1};
    D first[2] = {D::Hold, D::Hold};
    bool shared = false;                     // another of ours rested at its price while it did
};
struct Out { int lag; D a, b; long a_ev, b_ev; bool before_ack, shared; };
static std::vector<Out> g_out;

struct Opt { int max_lag; bool multi; bool defaults; bool nosig; };

static void run_one(uint64_t seed, const Opt& o) {
    std::mt19937_64 rng(seed);
    auto U = [&](uint64_t n) { return rng() % n; };
    CancelParams p;
    if (!o.defaults) {                       // the test's parameter sets: (a) off, X and T drawn
        static constexpr double Xs[] = {0.0, 30.0, 50.0, 75.0};
        static constexpr uint32_t Ts[] = {2, 3, 5, 10};
        p.queue_n = 0;
        p.abandon_pct = Xs[U(4)];
        p.aggr_prints_n = Ts[U(4)];
        p.passive_side_only = U(2) != 0;
        p.aggr_from_signal = U(4) != 0;
        p.far_touch_min_stable_ns = 50'000;
    }
    if (o.nosig) p.aggr_from_signal = false;
    World A(1, p), B(2, p);
    World* W[2] = {&A, &B};
    Nbbo nb{100000, 100100};
    for (World* w : W) {
        g_st[w->sym].tac = TacticShared{}; g_st[w->sym].round_lot = 100;
        g_rl[w->sym][0] = {100000, 300}; g_rl[w->sym][1] = {100500, 300};
        g_nbbo[w->sym] = nb;
        w->ct.set_symbol(w->sym, T, T);
        w->ct.set_tracking(w->sym, true);
        w->jt.arm(w->sym, true);
    }
    std::map<std::pair<int, PriceType>, std::vector<uint64_t>> lv;      // time priority
    std::unordered_map<uint64_t, Ord> od;
    uint64_t next_ref = 1;
    FeedTimeType ts = 1'000'000'000;
    long nev = 0;
    int lanes[2] = {0, 0}, pending = 0;
    const auto nosig = [](uint16_t, Side, const JoinSide&) {};
    const auto best = [&](int side) -> PriceType {
        PriceType b = 0;
        for (const auto& [k, v] : lv)
            if (k.first == side && !v.empty()) b = (b == 0 || (side ? k.second < b : k.second > b)) ? k.second : b;
        return b;
    };
    const auto own_at = [&](int side, PriceType px) {
        int n = 0;
        for (uint64_t r : lv[{side, px}]) n += od[r].ours;
        return n;
    };
    const auto finalize = [&](const Ord& x) {
        if (!x.ours) return;
        g_out.push_back({x.lag0, x.first[0], x.first[1], x.first_ev[0] < 0 ? -1 : x.first_ev[0] - x.ev_add,
                         x.first_ev[1] < 0 ? -1 : x.first_ev[1] - x.ev_add, x.lag >= 0, x.shared});
    };
    const auto record = [&](int i, const std::vector<Got>& got) {
        for (const Got& g : got) {
            const auto it = od.find(g.id);
            if (it == od.end()) { std::printf("decision on an order that has left\n"); std::exit(2); }
            Ord& x = it->second;
            if (x.first_ev[i] < 0) { x.first_ev[i] = nev; x.first[i] = g.d; }
        }
    };
    for (int step = 0; step < 500; ++step) {
        if (U(100) >= 15) ts += 1000 + U(200'000);
        for (World* w : W) w->got.clear();
        const int side = static_cast<int>(U(2));
        const Side sd = static_cast<Side>(side);
        const PriceType bst = best(side);
        const PriceType base = bst ? bst : (side ? PriceType{100500} : PriceType{100000});
        const int dir = side ? 1 : -1;                                  // away from the touch
        const unsigned r = static_cast<unsigned>(U(100));
        if (r < 22) {                                                   // a market order joins
            const bool improve = bst != 0 && U(100) < 12 && (side ? base > 100300 : base < 100200);
            const PriceType px = improve ? base - dir * T : base + dir * T * static_cast<PriceType>(U(4));
            const uint32_t sz = U(100) < 35 ? static_cast<uint32_t>(10 + U(85)) : static_cast<uint32_t>(100 + U(300));
            const uint64_t ref = next_ref++;
            Ord x; x.side = side; x.px = px; x.sz = sz;
            od[ref] = x; lv[{side, px}].push_back(ref);
            for (World* w : W) {
                w->qh[ref] = w->q[{side, px}].push(w->pool, ref, sz);
                w->jt.on_add(w->sym, sd, px, sz, bst, ts, nosig);
                w->ct.on_book_update(w->sym, BookEvent::Add, sd, px, ts, uint8_t{0}, w->cb());
            }
        } else if (r < 34) {                                            // one of ours joins: never improving
            if (!bst || lanes[side] >= 8) continue;
            const PriceType px = base + dir * T * static_cast<PriceType>(U(4));
            const int there = own_at(side, px);
            if (there && !o.multi) continue;
            const uint64_t ref = next_ref++;
            Ord x; x.side = side; x.px = px; x.ours = true;
            x.sz = static_cast<uint32_t>(50 * (1 + U(6)));
            x.lag = x.lag0 = static_cast<int>(1 + U(static_cast<uint64_t>(o.max_lag)));
            x.t_add = ts; x.ev_add = nev + 1;
            if (there) { x.shared = true; for (uint64_t y : lv[{side, px}]) if (od[y].ours) od[y].shared = true; }
            ++pending; ++lanes[side];
            od[ref] = x; lv[{side, px}].push_back(ref);
            Ord& me = od[ref];
            for (int i = 0; i < 2; ++i) {
                World* w = W[i];
                const uint32_t h = w->q[{side, px}].push(w->pool, ref, me.sz);
                w->qh[ref] = h;
                if (i == 0) me.h[0] = w->ct.on_place(ref, w->sym, sd, px, ts, h, w->cb());
                else        w->jt.on_add(w->sym, sd, px, me.sz, bst, ts, nosig);   // B: an order like any other
                w->ct.on_book_update(w->sym, BookEvent::Add, sd, px, ts, uint8_t{0}, w->cb());
            }
        } else if (r < 56 || (r >= 70 && r < 76)) {                     // an order leaves; 70-75: replaced
            std::vector<std::pair<int, PriceType>> keys;
            for (const auto& [k, v] : lv) if (!v.empty()) keys.push_back(k);
            if (keys.empty()) continue;
            const auto key = keys[U(keys.size())];
            std::vector<uint64_t>& L = lv[key];
            const std::size_t i = U(L.size());
            const uint64_t ref = L[i];
            const Ord x = od[ref];
            const bool replace = r >= 70 && !x.ours;
            static constexpr BookEvent kEv[3] = {BookEvent::Delete, BookEvent::Cancel, BookEvent::Replace};
            const BookEvent ev = replace ? BookEvent::Replace : U(100) < 40 ? BookEvent::Execute : kEv[U(3)];
            const Side xs = static_cast<Side>(x.side);
            L.erase(L.begin() + static_cast<std::ptrdiff_t>(i));
            if (x.ours) { finalize(x); if (x.lag >= 0) --pending; --lanes[x.side]; }
            od.erase(ref);
            PriceType npx = 0; uint64_t nref = 0; PriceType nbest = 0;
            if (replace) {                                              // its new order: another level, same side
                nbest = best(x.side);
                const PriceType nb0 = nbest ? nbest : (x.side ? PriceType{100500} : PriceType{100000});
                npx = nb0 + (x.side ? 1 : -1) * T * static_cast<PriceType>(U(4));
                nref = next_ref++;
                Ord n; n.side = x.side; n.px = npx; n.sz = x.sz;
                od[nref] = n; lv[{x.side, npx}].push_back(nref);
            }
            for (World* w : W) {
                const uint32_t h = w->qh[ref];
                const uint32_t sz0 = QBlockList::size_of(w->pool, h);
                const uint8_t f = w->q[key].remove(w->pool, h);
                if (f & kQTracked) w->ct.on_book_remove(w->sym, xs, h);
                if (ev == BookEvent::Execute) w->jt.on_execute(w->sym, xs, x.px, sz0);
                w->ct.on_book_update(w->sym, ev, xs, x.px, ts, f, w->cb());
                w->qh.erase(ref);
                if (replace) {
                    w->qh[nref] = w->q[{x.side, npx}].push(w->pool, nref, x.sz);
                    w->jt.on_add(w->sym, xs, npx, x.sz, nbest, ts, nosig);
                }
            }
        } else if (r < 66) {                                            // cut in place
            std::vector<uint64_t> refs;
            for (const auto& [k, v] : lv) for (uint64_t y : v) refs.push_back(y);
            if (refs.empty()) continue;
            const uint64_t ref = refs[U(refs.size())];
            Ord& x = od[ref];
            if (x.sz < 2) continue;
            const Side xs = static_cast<Side>(x.side);
            const uint32_t d = static_cast<uint32_t>(1 + U(x.sz - 1));
            const BookEvent ev = U(2) ? BookEvent::Execute : BookEvent::Cancel;
            x.sz -= d;
            for (World* w : W) {
                const uint8_t f = w->q[{x.side, x.px}].reduce(w->pool, w->qh[ref], d);
                if (ev == BookEvent::Execute) w->jt.on_execute(w->sym, xs, x.px, d);
                w->ct.on_book_update(w->sym, ev, xs, x.px, ts, f, w->cb());
            }
        } else if (r < 86) {                                            // the NBBO moves; seen by the next update
            nb.bid = 99800 + T * static_cast<PriceType>(U(5));
            nb.ask = nb.bid + T * static_cast<PriceType>(1 + U(4)) + (U(100) < 20 ? 20 * T : 0);
            if (side && U(100) < 20) nb.bid -= 20 * T;
            for (World* w : W) g_nbbo[w->sym] = nb;
            continue;
        } else if (r < 94) {                                            // an update that changes no queue
            for (World* w : W) w->ct.on_book_update(w->sym, BookEvent::Modify, Side::Buy, 0, ts, uint8_t{0}, w->cb());
        } else if (r < 97) {                                            // the strategy ends a signal, never
            if (pending) continue;                                      //   before the ack of an order placed on it
            for (World* w : W) { JoinTactic::kill(w->sym, sd); w->jt.poll(w->sym); }
            continue;
        } else {
            continue;
        }
        ++nev;
        record(0, A.got);
        record(1, B.got);
        for (auto& [ref, x] : od) {                                     // B learns of its orders: the acks
            if (x.lag < 0 || --x.lag > 0) continue;
            B.got.clear();
            CT::LateInfo li;
            x.h[1] = B.ct.on_place_late(ref, B.sym, static_cast<Side>(x.side), x.px, x.t_add, B.qh[ref], B.cb(), &li);
            x.lag = -1; --pending;
            if (x.h[1] == CT::kInvalid) { std::printf("on_place_late refused\n"); std::exit(2); }
            record(1, B.got);
        }
    }
    for (auto& [ref, x] : od) if (x.ours && x.lag < 0) finalize(x);     // still resting at the end
}

static const char* kName[5] = {"none", "(a)", "(b)", "(c)", "(d)"};

int main(int argc, char** argv) {
    const int streams = argc > 1 ? std::atoi(argv[1]) : 1000;
    Opt o{argc > 2 ? std::atoi(argv[2]) : 60, argc > 3 && std::atoi(argv[3]) != 0, argc > 4 ? std::atoi(argv[4]) != 0 : true,
          argc > 5 && std::atoi(argv[5]) != 0};
    if (streams < 1 || o.max_lag < 1) { std::printf("usage: late_approx [streams] [max lag] [multi] [defaults] [no seeding]\n"); return 2; }
    for (int i = 0; i < streams; ++i) run_one(9000 + static_cast<uint64_t>(i), o);
    const int edges[] = {1, 5, 13, 33, 61};
    std::printf("max lag %d, %s, %s parameters%s: %zu orders of ours\n", o.max_lag, o.multi ? "several of ours a price" : "one of ours a price",
                o.defaults ? "default" : "test", o.nosig ? ", (d) not seeded from the join signal" : "", g_out.size());
    for (int bkt = -1; bkt < 4; ++bkt) {
        const int lo = bkt < 0 ? 1 : edges[bkt], hi = bkt < 0 ? 1000 : edges[bkt + 1] - 1;
        std::size_t n = 0, left = 0;
        // A decided before the ack (in the gap): what B decides
        std::size_t g_n[5] = {}, g_at[5] = {}, g_later[5] = {}, g_other[5] = {}, g_never[5] = {};
        // A had not decided by the ack
        std::size_t m_n = 0, m_neither = 0, m_same[5] = {}, m_later[5] = {}, m_earlier[5] = {}, m_other[5] = {}, m_aonly[5] = {},
                    m_a[5] = {}, m_bonly[5] = {};
        std::vector<long> delay, adv;
        for (const Out& r : g_out) {
            if (r.lag < lo || r.lag > hi) continue;
            if (o.multi && !r.shared) continue;
            if (r.before_ack) { ++left; continue; }
            ++n;
            const int a = static_cast<int>(r.a), b = static_cast<int>(r.b);
            const long ack = r.lag - 1;
            if (a && r.a_ev <= ack) {
                ++g_n[a];
                if (b == a && r.b_ev == ack) ++g_at[a];
                else if (b == a) ++g_later[a];
                else if (b) ++g_other[a];
                else ++g_never[a];
                continue;
            }
            ++m_n;
            if (a) ++m_a[a];
            if (!a && !b) ++m_neither;
            else if (a == b && r.a_ev == r.b_ev) ++m_same[a];
            else if (a == b && r.b_ev > r.a_ev) { ++m_later[a]; delay.push_back(r.b_ev - r.a_ev); }
            else if (a == b) { ++m_earlier[a]; adv.push_back(r.a_ev - r.b_ev); }
            else if (a && b) ++m_other[a];
            else if (a) ++m_aonly[a];
            else ++m_bonly[b];
        }
        if (!n) continue;
        const auto med = [](std::vector<long>& v) { if (v.empty()) return -1L; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        std::size_t gt = 0; for (int a = 1; a < 5; ++a) gt += g_n[a];
        std::printf("lag %d-%d: %zu registered at the ack (%zu left the book before it)\n", lo, std::min(hi, o.max_lag), n, left);
        std::printf("  A decided before the ack: %zu\n", gt);
        for (int a = 1; a < 5; ++a)
            if (g_n[a]) std::printf("    %s %zu: B the same at the ack %zu, the same later %zu, another first %zu, never %zu\n", kName[a],
                                    g_n[a], g_at[a], g_later[a], g_other[a], g_never[a]);
        std::size_t ms = 0, ml = 0, me = 0, mo = 0, ma = 0, mb = 0;
        for (int a = 0; a < 5; ++a) { ms += m_same[a]; ml += m_later[a]; me += m_earlier[a]; mo += m_other[a]; ma += m_aonly[a]; mb += m_bonly[a]; }
        std::printf("  A undecided at the ack: %zu; neither decides %zu, the same decision on the same update %zu, the same later %zu "
                    "(median %ld updates), earlier %zu (median %ld), another %zu, A only %zu, B only %zu\n",
                    m_n, m_neither, ms, ml, med(delay), me, med(adv), mo, ma, mb);
        for (int a = 1; a < 5; ++a)
            if (m_a[a] || m_bonly[a]) std::printf("    A %s %zu: same %zu, later %zu, earlier %zu, another %zu, B never %zu; B %s where A never: %zu\n",
                                                  kName[a], m_a[a], m_same[a], m_later[a], m_earlier[a], m_other[a], m_aonly[a], kName[a], m_bonly[a]);
    }
    return 0;
}
