// mdsys_2_cancel_test.cpp — regression tests for md::CancelTactic (mdsys_2.hpp [4]),
// its four triggers (a) queue position, (b) at the front after the queue
// cancelled, (c) too deep and (d) out-competed, md::JoinTactic ([5]) and the
// queue flags in qblocklist.hpp, with the volatility taken from the NBBO
// (default) and from the venue's round-lot BBO (VolSource::Venue), and with
// size increases that keep priority (kSizeUp) and that do not.
//
// The suite runs against both instantiations — CancelTacticT<true> (AVX-512VL
// kernels) and CancelTacticT<false> (scalar reference) — then feeds both the
// same randomized stream of book updates (every event kind, venue order flow
// in per-level QBlockLists with their tracked/ahead flags, venue orders cut
// and grown in place, our orders entering, partly filled, cut and leaving the
// book, join signals killed and renewed with our orders placed at and around
// their prices, NBBO moves between updates, degenerate quotes, early
// timestamps, tracking pins, re-entrant callbacks, stale handles, both
// tactics' decode-stage prefetches on every event), for four
// parameter sets, and requires bit-identical decisions and state. That
// compares two implementations of one logic; the "model" tests compare the
// logic itself — (a), (b) and the join tactic — with brute-force models
// written from the requirements. QBlockList is checked against a reference
// model on its own. getNBBO is this file's: a
// per-symbol array the tests write, standing in for the host's snapshot.
// getVenueRoundLotBBO and getVenueRoundLot likewise, or the round-lot quotes
// of a small book the test builds; they count their calls. getSymbolState
// returns a per-symbol SymbolState array. Exit status 0 = all checks passed.
//
// Build:  g++ -std=c++23 -O2 -march=graniterapids -Wall -Wextra -I. mdsys_2_cancel_test.cpp -o cancel_test
// Debug:  g++ -std=c++23 -O1 -g -march=graniterapids -fsanitize=address,undefined -I. mdsys_2_cancel_test.cpp -o cancel_test

#include "mdsys_2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace md;

// ---- the host hook, test version ------------------------------------------------------------

static Nbbo     g_nbbo[64];                      // "current NBBO" per symbol; zero = both absent
static Nbbo     g_venue[64];                     // venue round-lot BBO when no MiniBook is attached
static uint64_t g_nbbo_reads = 0;
Nbbo md::getNBBO(uint16_t symst_idx) noexcept { ++g_nbbo_reads; return g_nbbo[symst_idx]; }

struct MiniBook;
static const MiniBook* g_vbook = nullptr;          // venue book behind getVenueRoundLotBBO, if any
static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...)                                                        \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            ++g_fail;                                                           \
            std::printf("    FAIL line %d: %s -- ", __LINE__, #cond);           \
            std::printf(__VA_ARGS__);                                           \
            std::printf("\n");                                                  \
        }                                                                       \
    } while (0)

static constexpr PriceType T = 100;              // $0.01 with 4 implied decimals

// SymbolState per symbol: the round lot and the join/cancel shared state.
static std::vector<SymbolState> g_st(64);
SymbolState& md::getSymbolState(uint16_t symst_idx) noexcept { return g_st[symst_idx]; }
static RlQuote g_rl[64][2];                      // getVenueRoundLot without a book
static uint64_t g_rl_reads = 0;

static void run(const char* tag, const char* name, void (*f)()) {
    std::fill_n(g_nbbo, 64, Nbbo{0, 0});
    std::fill_n(g_venue, 64, Nbbo{0, 0});
    for (auto& st : g_st) { st.tac = TacticShared{}; st.round_lot = 100; }
    for (auto& r : g_rl) r[0] = r[1] = RlQuote{0, 0};
    g_vbook = nullptr;
    const int before = g_fail;
    f();
    std::printf("%s  [%s] %s\n", g_fail == before ? "ok  " : "FAIL", tag, name);
}

// ---- a minimal venue book: one QBlockList per (symbol, side, price) level ---------------------
// The queues the host keeps in PrcLevelGeneric::plist, over one shared pool.

struct MiniBook {
    struct Ord { uint16_t sym; Side sd; PriceType px; uint32_t qh; };
    using Key = std::tuple<uint16_t, unsigned, PriceType>;
    QBlockPool pool;
    std::map<Key, QBlockList> lv;
    std::map<Key, uint64_t> agg;                 // level size (PrcLevelGeneric::agg_sz)
    std::unordered_map<uint64_t, Ord> ord;
    uint32_t round_lot = 100;

    explicit MiniBook(std::size_t nblocks = 4096) : pool(nblocks) {}
    QBlockList& level(uint16_t s, Side sd, PriceType px) { return lv[Key{s, unsigned(sd), px}]; }
    uint32_t add(uint64_t ref, uint16_t s, Side sd, PriceType px, uint32_t sz) {
        QBlockList& q = level(s, sd, px);
        const uint32_t qh = q.push(pool, ref, sz);
        if (qh == QBlockList::kBadHandle) { if (q.empty()) lv.erase(Key{s, unsigned(sd), px}); return qh; }
        ord[ref] = {s, sd, px, qh};
        agg[Key{s, unsigned(sd), px}] += sz;
        return qh;
    }
    // Round-lot BBO as LadderBook::roundlot_best computes it: from the best
    // level down, the price at which the cumulative size first reaches a
    // round lot; 0 when the side never does.
    Nbbo rl_bbo(uint16_t s) const {
        Nbbo r{0, 0};
        const auto first_ask = agg.lower_bound(Key{s, 1u, std::numeric_limits<PriceType>::min()});
        uint64_t cum = 0;
        for (auto it = first_ask; it != agg.begin(); ) {
            --it;
            if (std::get<0>(it->first) != s || std::get<1>(it->first) != 0u) break;
            if ((cum += it->second) >= round_lot) { r.bid = std::get<2>(it->first); break; }
        }
        cum = 0;
        for (auto it = first_ask; it != agg.end(); ++it) {
            if (std::get<0>(it->first) != s || std::get<1>(it->first) != 1u) break;
            if ((cum += it->second) >= round_lot) { r.ask = std::get<2>(it->first); break; }
        }
        return r;
    }
    // Delete or full execution. Returns QBlockList::remove's flags.
    uint8_t remove(uint64_t ref, Ord* out = nullptr) {
        const auto it = ord.find(ref);
        const Ord o = it->second;
        const Key k{o.sym, unsigned(o.sd), o.px};
        QBlockList& q = lv[k];
        if ((agg[k] -= QBlockList::size_of(pool, o.qh)) == 0) agg.erase(k);
        const uint8_t f = q.remove(pool, o.qh);
        if (q.empty()) lv.erase(k);
        ord.erase(it);
        if (out) *out = o;
        return f;
    }
    // Round-lot best of one side, with the size from the top through it.
    RlQuote rl_side(uint16_t s, Side sd) const {
        uint64_t cum = 0;
        if (sd == Side::Buy) {
            const auto first_ask = agg.lower_bound(Key{s, 1u, std::numeric_limits<PriceType>::min()});
            for (auto it = first_ask; it != agg.begin(); ) {
                --it;
                if (std::get<0>(it->first) != s || std::get<1>(it->first) != 0u) break;
                if ((cum += it->second) >= round_lot) return {std::get<2>(it->first), cum};
            }
        } else {
            for (auto it = agg.lower_bound(Key{s, 1u, std::numeric_limits<PriceType>::min()});
                 it != agg.end(); ++it) {
                if (std::get<0>(it->first) != s || std::get<1>(it->first) != 1u) break;
                if ((cum += it->second) >= round_lot) return {std::get<2>(it->first), cum};
            }
        }
        return {0, 0};
    }
    // Best price of a side (0 if empty).
    PriceType best(uint16_t s, Side sd) const {
        if (sd == Side::Buy) {
            const auto first_ask = agg.lower_bound(Key{s, 1u, std::numeric_limits<PriceType>::min()});
            if (first_ask == agg.begin()) return 0;
            const auto it = std::prev(first_ask);
            return (std::get<0>(it->first) == s && std::get<1>(it->first) == 0u) ? std::get<2>(it->first) : 0;
        }
        const auto it = agg.lower_bound(Key{s, 1u, std::numeric_limits<PriceType>::min()});
        return (it != agg.end() && std::get<0>(it->first) == s && std::get<1>(it->first) == 1u)
                   ? std::get<2>(it->first) : 0;
    }
    uint8_t reduce(uint64_t ref, uint32_t delta) {
        const Ord& o = ord.at(ref);
        agg[Key{o.sym, unsigned(o.sd), o.px}] -= delta;
        return level(o.sym, o.sd, o.px).reduce(pool, o.qh, delta);
    }
    // Size increase in place, keeping priority (MD_SIZE_UP_KEEPS_PRIORITY venues).
    uint8_t grow(uint64_t ref, uint32_t delta) {
        const Ord& o = ord.at(ref);
        agg[Key{o.sym, unsigned(o.sd), o.px}] += delta;
        return level(o.sym, o.sd, o.px).grow(pool, o.qh, delta);
    }
    uint32_t qh(uint64_t ref) const { return ord.at(ref).qh; }
};

static uint64_t g_venue_reads = 0;
Nbbo md::getVenueRoundLotBBO(uint16_t symst_idx) noexcept {
    ++g_venue_reads;
    return g_vbook ? g_vbook->rl_bbo(symst_idx) : g_venue[symst_idx];
}
RlQuote md::getVenueRoundLot(uint16_t symst_idx, Side sd) noexcept {
    ++g_rl_reads;
    return g_vbook ? g_vbook->rl_side(symst_idx, sd) : g_rl[symst_idx][static_cast<unsigned>(sd)];
}

static QBlockPool g_nopool(1);                   // for tests whose orders are not in a book

template <class CT>
struct Suite {
    using H = typename CT::Handle;
    using D = typename CT::Decision;
    static constexpr auto nop = [](H, uint64_t, D) {};
    static constexpr uint32_t NQ = CT::kNoQueue;

    struct Fired { H h; uint64_t id; D d; };
    struct Rec {
        std::vector<Fired> v;
        auto cb() { return [this](H h, uint64_t id, D d) { v.push_back({h, id, d}); }; }
    };

    // The NBBO becomes (bid, ask), then a non-execution book update arrives.
    template <class F>
    static void quote(CT& ct, uint16_t s, PriceType bid, PriceType ask, FeedTimeType ts, F&& cb) {
        g_nbbo[s] = {bid, ask};
        ct.on_book_update(s, BookEvent::Add, Side::Buy, bid, ts, 0, cb);
    }
    // A non-execution book update; the NBBO is whatever getNBBO returns now.
    template <class F>
    static void touch(CT& ct, uint16_t s, FeedTimeType ts, F&& cb) {
        ct.on_book_update(s, BookEvent::Modify, Side::Buy, 0, ts, 0, cb);
    }
    // An execution against a resting order on side `rest` at px.
    template <class F>
    static void exec(CT& ct, uint16_t s, Side rest, PriceType px, FeedTimeType ts, F&& cb) {
        ct.on_book_update(s, BookEvent::Execute, rest, px, ts, 0, cb);
    }
    // Our order enters the book (the Add carrying our ref), then the tactic tracks it.
    static H place(CT& ct, MiniBook& bk, uint64_t id, uint16_t s, Side sd, PriceType px,
                   uint32_t sz, FeedTimeType ts) {
        return ct.on_place(id, s, sd, px, ts, bk.add(id, s, sd, px, sz), nop);
    }
    template <class F>
    static H place_cb(CT& ct, MiniBook& bk, uint64_t id, uint16_t s, Side sd, PriceType px,
                      uint32_t sz, FeedTimeType ts, F&& cb) {
        return ct.on_place(id, s, sd, px, ts, bk.add(id, s, sd, px, sz), cb);
    }
    // An order leaves the book the way the host reports it: the queue's flags,
    // on_book_remove for one of ours, then the book update carrying the flags.
    template <class F>
    static uint8_t leave(CT& ct, MiniBook& bk, uint64_t ref, BookEvent ev, FeedTimeType ts, F&& cb) {
        MiniBook::Ord o;
        const uint8_t f = bk.remove(ref, &o);
        if (f & kQTracked) ct.on_book_remove(o.sym, o.sd, o.qh);
        ct.on_book_update(o.sym, ev, o.sd, o.px, ts, f, cb);
        return f;
    }

    // Buy 2 ticks behind a 10.00/10.01 market, tau 1 ms (symbol pinned for
    // the warm-up, then tracked through the order), then the offer gaps out
    // to 10.10, first seen by the book update at time u. Returns the order.
    static H wide_offer_setup(CT& ct, FeedTimeType& u, Rec& r) {
        ct.set_symbol(1, T, T);                                     // hist spread: 1 tick
        u = 2'000'000'000;
        ct.set_tracking(1, true);
        quote(ct, 1, 100000, 100100, u, nop);
        for (int k = 0; k < 30; ++k) { u += 1'000'000; exec(ct, 1, Side::Buy, 100000, u, nop); }
        const H h = ct.on_place(2, 1, Side::Buy, 99800, u, NQ, nop);
        ct.set_tracking(1, false);                                  // tracked by the order now
        u += 1'000'000;
        quote(ct, 1, 100000, 101000, u, r.cb());
        return h;
    }

    // ---- baseline -------------------------------------------------------------

    static void baseline_depth_fires_once_far_touch_settles() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        FeedTimeType t = 1'000'000'000;
        quote(ct, 1, 1000000, 1000100, t, nop);                         // untracked yet: not seen
        const H h = ct.on_place(7, 1, Side::Buy, 1000000, t, NQ, nop);       // at the bid
        for (int i = 0; i < 10; ++i) { t += 50'000'000; exec(ct, 1, Side::Sell, 1000100, t, nop); }  // tau 50 ms
        Rec r; int fired_at = -1;
        for (int i = 1; i <= 20 && r.v.empty(); ++i) {
            t += 100'000'000;
            const PriceType b = 1000000 + 2 * T * i;                      // +2 ticks / 100 ms
            quote(ct, 1, b, b + T, t, r.cb());
            CHECK(r.v.empty(), "fired on the update that moved the far touch (step %d)", i);
            touch(ct, 1, t + 1'000'000, r.cb());                           // next book update, 1 ms on
            if (!r.v.empty()) fired_at = i;
        }
        CHECK(r.v.size() == 1 && r.v[0].h == h && r.v[0].d == D::CancelDepth, "n=%zu", r.v.size());
        CHECK(fired_at == 3, "fired at step %d, expected 3 (depth 7 ticks > 5-tick floor)", fired_at);
        CHECK(!ct.tracking(1), "symbol still tracked with its only order decided");
    }

    static void baseline_outcompeted_counts_only_executions_against_our_side() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        FeedTimeType t = 1'000'000'000;
        g_nbbo[1] = {1004000, 1004100};
        const H a = ct.on_place(8, 1, Side::Buy, 1003700, t, NQ, nop);         // 3 ticks back
        Rec r;
        for (int i = 0; i < 12; ++i) { t += 1'000'000; exec(ct, 1, Side::Sell, 1004100, t, r.cb()); } // offers lifted
        CHECK(r.v.empty() && ct.order(a)->n_aggr_prints == 0,
              "buyer-initiated executions counted: %u", ct.order(a)->n_aggr_prints);
        for (int i = 0; i < 10; ++i) { t += 1'000'000; exec(ct, 1, Side::Buy, 1004000, t, r.cb()); }  // bids hit
        CHECK(r.v.size() == 1 && r.v[0].h == a && r.v[0].d == D::CancelOutcompeted, "n=%zu", r.v.size());
    }

    static void baseline_reentry_hysteresis() {
        CT ct(4, 16, g_nopool); FeedTimeType u; Rec r;
        wide_offer_setup(ct, u, r);
        touch(ct, 1, u + 300'000, r.cb());
        CHECK(r.v.size() == 1, "setup: depth cancel did not fire");
        CHECK(!ct.replace_allowed(1, Side::Buy, 99000, u + 1'000'000), "20 ticks deep allowed right after cancel");
        CHECK(ct.replace_allowed(1, Side::Buy, 100300, u + 1'000'000), "depth at 0.7x threshold refused");
        CHECK(ct.replace_allowed(1, Side::Buy, 99000, u + 301'000'000'000ull), "not aged out after reset window");
    }

    // The execution pass on the sell side, with no order of ours on the buy
    // side: (d) counts offers lifted below our sell and nothing else, and (c)
    // for a sell is decided on an execution update as on any other.
    static void execution_pass_on_the_sell_side() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        FeedTimeType t = 1'000'000'000;
        g_nbbo[1] = {1004000, 1004100};
        const H a = ct.on_place(8, 1, Side::Sell, 1004400, t, NQ, nop);        // 3 ticks behind the offer
        Rec r;
        for (int i = 0; i < 12; ++i) { t += 1'000'000; exec(ct, 1, Side::Buy, 1004000, t, r.cb()); }   // bids hit
        CHECK(r.v.empty() && ct.order(a)->n_aggr_prints == 0,
              "seller-initiated executions counted for a sell: %u", ct.order(a)->n_aggr_prints);
        for (int i = 0; i < 9; ++i) { t += 1'000'000; exec(ct, 1, Side::Sell, 1004100, t, r.cb()); }   // offers lifted
        CHECK(r.v.empty() && ct.order(a)->n_aggr_prints == 9, "count %u", ct.order(a)->n_aggr_prints);
        t += 1'000'000; exec(ct, 1, Side::Sell, 1004400, t, r.cb());           // at our own price: not below it
        CHECK(r.v.empty() && ct.order(a)->n_aggr_prints == 9, "an execution at our price counted");
        t += 1'000'000; exec(ct, 1, Side::Sell, 1004300, t, r.cb());
        CHECK(r.v.size() == 1 && r.v[0].h == a && r.v[0].d == D::CancelOutcompeted, "n=%zu", r.v.size());

        CT c2(4, 16, g_nopool); c2.set_symbol(1, T, T);                        // (c): the bid gaps away from a sell
        FeedTimeType u = 2'000'000'000;
        c2.set_tracking(1, true);
        quote(c2, 1, 100000, 100100, u, nop);
        for (int k = 0; k < 30; ++k) { u += 1'000'000; exec(c2, 1, Side::Buy, 100000, u, nop); }   // tau 1 ms
        const H b = c2.on_place(3, 1, Side::Sell, 100300, u, NQ, nop);
        c2.set_tracking(1, false);
        u += 1'000'000;
        Rec q;
        quote(c2, 1, 99100, 100100, u, q.cb());                                // first seen here: never fires itself
        CHECK(q.v.empty(), "fired on the update that moved the far touch");
        exec(c2, 1, Side::Buy, 99100, u + 300'000, q.cb());                    // a bid hit, the gate open
        CHECK(q.v.size() == 1 && q.v[0].h == b && q.v[0].d == D::CancelDepth, "n=%zu", q.v.size());
    }

    // One execution on which both (d) and (c) hold for an order: decided
    // once, as out-competed, and (c) is not evaluated for it — the re-entry
    // hysteresis, which only a depth cancel sets, stays clear.
    static void execution_that_completes_d_skips_c() {
        CancelParams p; p.aggr_prints_n = 1;
        CT ct(4, 16, g_nopool, p); FeedTimeType u; Rec r;
        const H h = wide_offer_setup(ct, u, r);                                // deep, far touch moved at u
        CHECK(r.v.empty() && ct.evaluate(h) == D::Hold, "setup: decided before the gate opened");
        exec(ct, 1, Side::Buy, 100000, u + 300'000, r.cb());                   // a bid above ours is hit; gate open
        CHECK(r.v.size() == 1 && r.v[0].h == h && r.v[0].d == D::CancelOutcompeted, "n=%zu d=%d", r.v.size(),
              r.v.empty() ? -1 : int(r.v[0].d));
        CHECK(ct.replace_allowed(1, Side::Buy, 99000, u + 1'000'000), "hysteresis armed by a (d) cancel");
    }

    // The threshold recorded for the re-entry hysteresis is the one (c)
    // fired against, volatility term included. With sigma * sqrt(horizon)
    // far above the cheap floors, re-entry is allowed up to 0.7 x that
    // threshold and no deeper — whether the cancel was decided on a book
    // update or on an execution.
    static void hysteresis_uses_the_threshold_that_fired() {
        for (int on_exec = 0; on_exec < 2; ++on_exec) {
            CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
            FeedTimeType t = 1'000'000'000;
            ct.set_tracking(1, true);
            PriceType bid = 1'000'000;
            quote(ct, 1, bid, bid + T, t, nop);
            for (int k = 0; k < 400; ++k) {                                    // the mid moves 4 ticks every ms;
                t += 1'000'000;                                                //   trades 50 ms apart
                bid += (k & 1) ? 4 * T : -4 * T;
                quote(ct, 1, bid, bid + T, t, nop);
                if (k % 50 == 0) exec(ct, 1, Side::Sell, bid + T, t, nop);
            }
            const PriceType thr0 = ct.threshold(1);
            const H h = ct.on_place(5, 1, Side::Buy, bid + T - (thr0 - T), t, NQ, nop);   // one tick inside thr
            ct.set_tracking(1, false);
            Rec r;
            t += 1'000'000;
            const PriceType ask = bid + 7 * T;
            quote(ct, 1, bid, ask, t, r.cb());                                 // the offer moves 6 ticks away
            const PriceType thr = ct.threshold(1), depth = ask - ct.order(h)->px;
            CHECK(r.v.empty() && thr > 20 * T && depth > thr,
                  "setup: thr %lld, depth %lld, %zu decisions", (long long)thr, (long long)depth, r.v.size());
            t += 300'000;                                                      // the far touch has settled
            if (on_exec) exec(ct, 1, Side::Sell, ask, t, r.cb());              // an offer lifted: no (d) for a buy
            else         touch(ct, 1, t, r.cb());
            CHECK(r.v.size() == 1 && r.v[0].h == h && r.v[0].d == D::CancelDepth, "on_exec %d: n=%zu", on_exec,
                  r.v.size());
            const PriceType lim = static_cast<PriceType>(0.7 * static_cast<double>(thr));
            CHECK(ct.replace_allowed(1, Side::Buy, ask - (lim - T), t + 1'000),
                  "on_exec %d: re-entry refused inside 0.7 x the threshold that fired (%lld)", on_exec, (long long)thr);
            CHECK(!ct.replace_allowed(1, Side::Buy, ask - (lim + 2 * T), t + 1'000),
                  "on_exec %d: re-entry allowed beyond 0.7 x the threshold that fired", on_exec);
        }
    }

    static void lane_capacity_per_side() {
        CT ct(4, 64, g_nopool); ct.set_symbol(1, T, 0);
        int ok = 0;
        for (unsigned i = 0; i < CT::kLanes; ++i)
            ok += ct.on_place(i + 1, 1, Side::Buy, 99000 - PriceType(i) * T, 1, NQ, nop) != CT::kInvalid;
        CHECK(ok == int(CT::kLanes), "placed %d of %u", ok, CT::kLanes);
        CHECK(ct.on_place(99, 1, Side::Buy, 98000, 1, NQ, nop) == CT::kInvalid && ct.place_failures() == 1,
              "side over capacity accepted");
        CHECK(ct.on_place(100, 1, Side::Sell, 101000, 1, NQ, nop) != CT::kInvalid, "other side blocked");
        CHECK(ct.on_place(101, 2, Side::Buy, 99000, 1, NQ, nop) != CT::kInvalid, "other symbol blocked");
    }

    // ---- 1. trigger: book updates and getNBBO --------------------------------------

    static void every_book_event_kind_triggers_the_check() {
        const BookEvent kinds[] = {BookEvent::Add, BookEvent::Delete, BookEvent::Modify,
                                   BookEvent::Cancel, BookEvent::Replace, BookEvent::Execute};
        int fired = 0;
        for (BookEvent k : kinds) {
            CT ct(4, 16, g_nopool); FeedTimeType u; Rec r;
            const H h = wide_offer_setup(ct, u, r);
            ct.on_book_update(1, k, Side::Sell, 101000, u + 300'000, 0, r.cb());   // Execute: an offer lifted
            fired += r.v.size() == 1 && r.v[0].h == h && r.v[0].d == D::CancelDepth;
        }
        CHECK(fired == 6, "(c) fired on %d of 6 event kinds", fired);
    }

    static void only_executions_count_toward_b() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        FeedTimeType t = 1'000'000'000;
        g_nbbo[1] = {100000, 100100};
        const H h = ct.on_place(1, 1, Side::Buy, 99500, t, NQ, nop);
        const BookEvent kinds[] = {BookEvent::Add, BookEvent::Delete, BookEvent::Modify,
                                   BookEvent::Cancel, BookEvent::Replace};
        for (int rep = 0; rep < 3; ++rep)
            for (BookEvent k : kinds) { t += 1'000; ct.on_book_update(1, k, Side::Buy, 100000, t, 0, nop); }
        CHECK(ct.order(h)->n_aggr_prints == 0, "non-execution updates counted: %u", ct.order(h)->n_aggr_prints);
        t += 1'000; exec(ct, 1, Side::Buy, 100000, t, nop);
        CHECK(ct.order(h)->n_aggr_prints == 1, "n=%u", ct.order(h)->n_aggr_prints);
    }

    // The far touch's change instant is when an update first SEES it; updates
    // that read the same pair (or a near-side-only change) do not restart it.
    static void unchanged_far_touch_does_not_restart_the_gate() {
        CT ct(4, 16, g_nopool); FeedTimeType u; Rec r;
        const H h = wide_offer_setup(ct, u, r);
        for (int k = 1; k <= 4; ++k) touch(ct, 1, u + FeedTimeType(k) * 40'000, r.cb());   // same pair
        g_nbbo[1].bid = 100050;                                            // near side only
        touch(ct, 1, u + 180'000, r.cb());
        CHECK(r.v.empty(), "fired inside the 200 us window");
        touch(ct, 1, u + 210'000, r.cb());
        CHECK(r.v.size() == 1 && r.v[0].h == h, "window restarted by an unchanged far touch (n=%zu)", r.v.size());
    }

    // getNBBO returning a crossed or one-sided quote holds (c) even with the
    // far touch itself stable; a far touch that goes absent and comes back
    // restarts the window.
    static void degenerate_nbbo_holds() {
        {
            CT ct(4, 16, g_nopool); FeedTimeType u; Rec r;
            const H h = wide_offer_setup(ct, u, r);                        // offer 10.10 since u
            g_nbbo[1] = {101100, 101000}; touch(ct, 1, u + 300'000, r.cb()); // crossed by the bid
            g_nbbo[1] = {0, 101000};      touch(ct, 1, u + 400'000, r.cb()); // bid absent
            CHECK(r.v.empty(), "fired on a crossed / one-sided NBBO (n=%zu)", r.v.size());
            g_nbbo[1] = {100000, 101000}; touch(ct, 1, u + 500'000, r.cb()); // valid again, offer unchanged
            CHECK(r.v.size() == 1 && r.v[0].h == h && r.v[0].d == D::CancelDepth, "n=%zu", r.v.size());
        }
        {
            CT ct(4, 16, g_nopool); FeedTimeType u; Rec r;
            const H h = wide_offer_setup(ct, u, r);
            g_nbbo[1] = {100000, 0};      touch(ct, 1, u + 300'000, r.cb()); // offer absent
            g_nbbo[1] = {100000, 101000}; touch(ct, 1, u + 400'000, r.cb()); // back: a new far touch
            touch(ct, 1, u + 550'000, r.cb());
            CHECK(r.v.empty(), "fired within 200 us of the far touch returning (n=%zu)", r.v.size());
            touch(ct, 1, u + 700'000, r.cb());
            CHECK(r.v.size() == 1 && r.v[0].h == h, "n=%zu", r.v.size());
        }
    }

    // Mirror of the buy case: a sell's far touch is the bid, so a bid-only
    // change must open the window and move the estimators.
    static void sell_side_far_touch_is_the_bid() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        FeedTimeType u = 2'000'000'000;
        ct.set_tracking(1, true);
        quote(ct, 1, 100000, 100100, u, nop);
        for (int k = 0; k < 30; ++k) { u += 1'000'000; exec(ct, 1, Side::Sell, 100100, u, nop); }
        const H h = ct.on_place(2, 1, Side::Sell, 100300, u, NQ, nop);         // 2 ticks above the offer
        ct.set_tracking(1, false);
        const double sig0 = ct.sigma_rate(1);
        Rec r;
        u += 1'000'000;
        quote(ct, 1, 99100, 100100, u, r.cb());                            // bid gaps down to 9.91
        CHECK(ct.sigma_rate(1) > sig0, "bid-only change left sigma at %.4g", ct.sigma_rate(1));
        touch(ct, 1, u + 100'000, r.cb());
        CHECK(r.v.empty(), "fired 100 us after the bid moved (window is 200 us)");
        touch(ct, 1, u + 300'000, r.cb());
        CHECK(r.v.size() == 1 && r.v[0].h == h && r.v[0].d == D::CancelDepth, "n=%zu", r.v.size());
    }

    // on_place reads getNBBO for its baseline: an offer that gapped away before
    // placement is the market the order joined, not a move after it.
    static void on_place_baseline_is_a_fresh_nbbo() {
        int fired[2] = {0, 0};
        for (int fresh = 0; fresh < 2; ++fresh) {
            CancelParams p; p.vol_horizon_cap_ns = 1;                   // isolate the moved-away clause
            CT ct(2, 8, g_nopool, p); ct.set_symbol(1, T, 0);
            auto cb = [&](H, uint64_t, D) { ++fired[fresh]; };
            FeedTimeType t = 1'000'000'000;
            g_nbbo[1] = fresh ? Nbbo{100000, 101000}                     // gapped; no book update yet
                              : Nbbo{100000, 100100};                    // control: the earlier pair
            ct.on_place(1, 1, Side::Buy, 99800, t + 1'000'000, NQ, nop);
            g_nbbo[1] = {100000, 101000};
            touch(ct, 1, t + 1'100'000, cb);                             // nothing moved since placement
            touch(ct, 1, t + 1'400'000, cb);
        }
        CHECK(fired[0] == 1, "control: a stale baseline should have (wrongly) fired, got %d", fired[0]);
        CHECK(fired[1] == 0, "fresh baseline fired %d on an unmoved market", fired[1]);
    }

    static void replace_allowed_reads_a_fresh_nbbo() {
        CT ct(4, 16, g_nopool); FeedTimeType u; Rec r;
        wide_offer_setup(ct, u, r);
        touch(ct, 1, u + 300'000, r.cb());
        CHECK(r.v.size() == 1, "setup: depth cancel did not fire");
        CHECK(!ct.replace_allowed(1, Side::Buy, 99800, u + 400'000), "re-entry allowed while still deep");
        g_nbbo[1] = {99700, 99800};                                        // market falls to us, unseen by updates
        CHECK(ct.replace_allowed(1, Side::Buy, 99700, u + 400'000), "re-entry refused against the current NBBO");
    }

    // ---- 2. timestamps ------------------------------------------------------------

    static void early_stamped_execution_is_clamped() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T); ct.set_tracking(1, true);
        FeedTimeType t = 1'000'000'000;
        quote(ct, 1, 100000, 100100, t, nop);
        for (int k = 0; k < 30; ++k) { t += 1'000'000; exec(ct, 1, Side::Buy, 100000, t, nop); }
        CHECK(ct.trade_gap_ns(1) == 1e6, "setup: gap EWMA %.6g", ct.trade_gap_ns(1));
        exec(ct, 1, Side::Buy, 100000, t - 5'000, nop);                    // stamped 5 us early: same instant
        exec(ct, 1, Side::Buy, 100000, t + 1'000'000, nop);
        CHECK(ct.trade_gap_ns(1) == 1e6, "gap EWMA %.6g after an early stamp, true 1e6", ct.trade_gap_ns(1));
        CHECK(ct.sym_time(1) == t + 1'000'000, "symbol clock moved backwards or skipped");
    }

    static void early_stamped_quote_keeps_sigma() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T); ct.set_tracking(1, true);
        FeedTimeType t = 1'000'000'000;
        PriceType b = 100000;
        auto walk = [&](int n) {
            for (int k = 0; k < n; ++k) { t += 100'000'000; b += (k & 1) ? T : -T; quote(ct, 1, b, b + T, t, nop); }
        };
        auto sig = [&] { return ct.sigma_rate(1) * std::sqrt(1e9) / T; };  // ticks / sqrt(s)
        walk(600);
        const double steady = sig();
        quote(ct, 1, b + T, b + 2 * T, t - 1'000, nop);                    // stamped 1 us early
        walk(600);
        CHECK(std::fabs(sig() / steady - 1.0) < 0.25, "sigma steady %.3f, 1 min after %.4f", steady, sig());
    }

    static void gate_holds_for_update_stamped_before_far_move() {
        CT ct(4, 16, g_nopool); FeedTimeType u; Rec r;
        const H h = wide_offer_setup(ct, u, r);
        CHECK(r.v.empty(), "fired on the update that moved the far touch");
        touch(ct, 1, u - 1'000, r.cb());                                    // stamped 1 us before it
        CHECK(r.v.empty(), "update stamped before the move opened the gate (n=%zu)", r.v.size());
        touch(ct, 1, u + 100'000, r.cb());
        CHECK(r.v.empty(), "fired 100 us after the move (window is 200 us)");
        CHECK(ct.evaluate(h) == D::Hold, "evaluate inside the window");
        touch(ct, 1, u + 300'000, r.cb());
        CHECK(r.v.size() == 1 && r.v[0].h == h && r.v[0].d == D::CancelDepth,
              "did not fire once the far touch held (n=%zu)", r.v.size());
        CHECK(ct.evaluate(h) == D::CancelDepth, "evaluate after the window");
    }

    static void outcompeted_ignores_executions_at_or_before_placement() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        FeedTimeType t = 1'000'000'000;
        g_nbbo[1] = {100000, 100100};
        const H h = ct.on_place(3, 1, Side::Buy, 99700, t + 10'000, NQ, nop);
        exec(ct, 1, Side::Buy, 100000, t + 5'000, nop);                    // executed before we existed
        exec(ct, 1, Side::Buy, 100000, t + 10'000, nop);                   // same instant as placement
        CHECK(ct.order(h)->n_aggr_prints == 0, "counted %u pre-placement executions", ct.order(h)->n_aggr_prints);
        exec(ct, 1, Side::Buy, 100000, t + 20'000, nop);
        CHECK(ct.order(h)->n_aggr_prints == 1, "n=%u", ct.order(h)->n_aggr_prints);
        // Placement is compared with the execution's own stamp, not the
        // symbol clock: an order stamped behind the clock does not count an
        // execution stamped before it, even when that arrives later.
        const H g = ct.on_place(4, 1, Side::Buy, 99700, t + 15'000, NQ, nop);  // clock is at t + 20 us
        exec(ct, 1, Side::Buy, 100000, t + 12'000, nop);                   // stamped before g existed
        CHECK(ct.order(g)->n_aggr_prints == 0, "counted on the clamped clock: n=%u", ct.order(g)->n_aggr_prints);
        exec(ct, 1, Side::Buy, 100000, t + 30'000, nop);
        CHECK(ct.order(g)->n_aggr_prints == 1, "n=%u", ct.order(g)->n_aggr_prints);
    }

    static void hysteresis_with_older_clock_does_not_age_out() {
        CT ct(4, 16, g_nopool); FeedTimeType u; Rec r;
        wide_offer_setup(ct, u, r);
        touch(ct, 1, u + 300'000, r.cb());
        CHECK(r.v.size() == 1, "setup: depth cancel did not fire");
        CHECK(!ct.replace_allowed(1, Side::Buy, 99000, u), "timestamp older than the cancel read as aged out");
    }

    // ---- 3. (d) from the resting side ------------------------------------------------

    static void level_clearing(bool nbbo_first) {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        FeedTimeType t = 1'000'000'000;
        g_nbbo[1] = {100000, 100100};                                      // 10.00 / 10.01
        const H h = ct.on_place(1, 1, Side::Buy, 99000, t, NQ, nop);            // 9.90
        Rec r; int fired_at = -1;
        for (int k = 0; k < 12; ++k) {
            t += 1'000'000;
            const PriceType bid = 100000 - k * T;                          // each sell clears the bid level
            if (nbbo_first) { g_nbbo[1] = {bid - T, bid + T}; exec(ct, 1, Side::Buy, bid, t, r.cb()); }
            else            { exec(ct, 1, Side::Buy, bid, t, r.cb()); quote(ct, 1, bid - T, bid + T, t, r.cb()); }
            if (fired_at < 0 && !r.v.empty()) fired_at = k;
        }
        CHECK(r.v.size() == 1 && r.v[0].h == h && r.v[0].d == D::CancelOutcompeted, "n=%zu", r.v.size());
        CHECK(fired_at == 9, "fired at sell %d, expected 9 (10th execution above 9.90)", fired_at);
    }
    static void level_clearing_nbbo_already_moved() { level_clearing(true); }
    static void level_clearing_nbbo_moves_after() { level_clearing(false); }

    static void sweep_in_one_timestamp_counts_each_execution() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        FeedTimeType t = 1'000'000'000;
        g_nbbo[1] = {100000, 100100};
        const H h = ct.on_place(1, 1, Side::Buy, 99500, t, NQ, nop);
        t += 1'000'000;
        g_nbbo[1] = {99800, 100100};                                       // post-sweep NBBO
        exec(ct, 1, Side::Buy, 100000, t, nop);
        exec(ct, 1, Side::Buy,  99900, t, nop);
        CHECK(ct.order(h)->n_aggr_prints == 2, "counted %u of 2", ct.order(h)->n_aggr_prints);
    }

    // The resting side decides the aggressor whatever the NBBO looks like.
    static void aggressor_from_resting_side_with_locked_or_crossed_nbbo() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        FeedTimeType t = 1'000'000'000;
        g_nbbo[1] = {100000, 100100};
        const H b = ct.on_place(1, 1, Side::Buy,   99500, t, NQ, nop);
        const H s = ct.on_place(2, 1, Side::Sell, 100500, t, NQ, nop);
        quote(ct, 1, 100000, 100000, t + 1'000'000, nop);                 // locked
        exec(ct, 1, Side::Buy, 100000, t + 2'000'000, nop);                // a bid hit
        CHECK(ct.order(b)->n_aggr_prints == 1 && ct.order(s)->n_aggr_prints == 0,
              "locked: buy %u sell %u", ct.order(b)->n_aggr_prints, ct.order(s)->n_aggr_prints);
        quote(ct, 1, 100100, 100000, t + 3'000'000, nop);                 // crossed
        exec(ct, 1, Side::Sell, 100100, t + 4'000'000, nop);               // an offer lifted
        CHECK(ct.order(b)->n_aggr_prints == 1 && ct.order(s)->n_aggr_prints == 1,
              "crossed: buy %u sell %u", ct.order(b)->n_aggr_prints, ct.order(s)->n_aggr_prints);
    }

    static void both_sides_count_without_passive_side_only() {
        for (int passive = 0; passive < 2; ++passive) {
            CancelParams p; p.passive_side_only = passive != 0;
            CT ct(4, 16, g_nopool, p); ct.set_symbol(1, T, T);
            FeedTimeType t = 1'000'000'000;
            g_nbbo[1] = {100000, 100100};
            const H b = ct.on_place(1, 1, Side::Buy,   99700, t, NQ, nop);
            const H s = ct.on_place(2, 1, Side::Sell, 100400, t, NQ, nop);
            exec(ct, 1, Side::Sell, 100100, t + 1'000, nop);               // offer lifted at 10.01
            exec(ct, 1, Side::Buy,  100000, t + 2'000, nop);               // bid hit at 10.00
            const uint32_t nb = ct.order(b)->n_aggr_prints, ns = ct.order(s)->n_aggr_prints;
            if (passive) CHECK(nb == 1 && ns == 1, "passive-side only: buy %u sell %u, expected 1/1", nb, ns);
            else         CHECK(nb == 2 && ns == 2, "both sides: buy %u sell %u, expected 2/2", nb, ns);
        }
    }

    // ---- 4. callback re-entrancy --------------------------------------------------

    static void callback_cancel_replace_does_not_double_count() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        FeedTimeType t = 1'000'000'000;
        g_nbbo[1] = {100000, 100100};
        const H a = ct.on_place(10, 1, Side::Buy, 99500, t, NQ, nop);
        for (int k = 0; k < 9; ++k) { t += 1'000; exec(ct, 1, Side::Buy, 100000, t, nop); }   // a at 9
        const H b = ct.on_place(11, 1, Side::Buy, 99600, t, NQ, nop);
        int calls = 0;
        auto cb = [&](H h, uint64_t id, D) {
            ++calls;
            CHECK(ct.on_done(h), "on_done from callback failed");
            CHECK(ct.on_place(id + 100, 1, Side::Buy, 99800, t, NQ, nop) != CT::kInvalid, "replace failed");
        };
        t += 1'000; exec(ct, 1, Side::Buy, 100000, t, cb);
        CHECK(calls == 1, "calls=%d", calls);
        CHECK(ct.order(b)->n_aggr_prints == 1, "sibling counted %u for one execution", ct.order(b)->n_aggr_prints);
        CHECK(!ct.order(a), "canceled order still resolves");
    }

    static void callback_freeing_pending_sibling_drops_its_decision() {
        CT ct(4, 3, g_nopool); ct.set_symbol(1, T, T);
        FeedTimeType t = 1'000'000'000;
        g_nbbo[1] = {100000, 100100};
        H hs[3];
        for (int i = 0; i < 3; ++i) hs[i] = ct.on_place(20 + i, 1, Side::Buy, 99500 - i * T, t, NQ, nop);
        for (int k = 0; k < 9; ++k) { t += 1'000; exec(ct, 1, Side::Buy, 100000, t, nop); }
        int calls = 0;
        auto cb = [&](H, uint64_t, D) { ++calls; for (H h : hs) ct.on_done(h); };
        t += 1'000; exec(ct, 1, Side::Buy, 100000, t, cb);                 // all three reach 10
        CHECK(calls == 1, "calls=%d: decisions for orders freed by an earlier callback must drop", calls);
        int placed = 0;
        for (int i = 0; i < 3; ++i) placed += ct.on_place(30 + i, 1, Side::Buy, 99000, t, NQ, nop) != CT::kInvalid;
        CHECK(placed == 3 && ct.on_place(99, 1, Side::Buy, 99000, t, NQ, nop) == CT::kInvalid,
              "free lanes corrupted (placed %d of 3)", placed);
    }

    static void nested_update_for_another_symbol_from_callback() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T); ct.set_symbol(2, T, T);
        FeedTimeType t = 1'000'000'000;
        g_nbbo[1] = {100000, 100100};
        g_nbbo[2] = {200000, 200100};
        const H a = ct.on_place(1, 1, Side::Buy,  99500, t, NQ, nop);
        const H b = ct.on_place(2, 2, Side::Buy, 199500, t, NQ, nop);
        for (int k = 0; k < 9; ++k) {
            t += 1'000;
            exec(ct, 1, Side::Buy, 100000, t, nop);
            exec(ct, 2, Side::Buy, 200000, t, nop);
        }
        std::vector<H> got;
        struct Cb {
            CT& ct; std::vector<H>& got; FeedTimeType t;
            void operator()(H h, uint64_t, D) {
                got.push_back(h);
                if (got.size() == 1) ct.on_book_update(2, BookEvent::Execute, Side::Buy, 200000, t, 0, *this);
            }
        };
        t += 1'000;
        Cb cb{ct, got, t};
        exec(ct, 1, Side::Buy, 100000, t, cb);
        CHECK(got.size() == 2 && got[0] == a && got[1] == b, "delivered %zu decisions", got.size());
    }

    // A decision's callback feeds the same symbol a newer update. The outer
    // update's estimator work (its quote change and its execution) was
    // deferred past delivery; the nested update must complete it first, at
    // the outer update's clock, so the result equals the two updates in
    // sequence — bit for bit. (Pinned, so the symbol stays tracked after its
    // only order is decided.)
    static void nested_same_symbol_update_completes_deferred_bookkeeping() {
        CT a(4, 16, g_nopool), c(4, 16, g_nopool); FeedTimeType ua, uc; Rec ra, rc;
        wide_offer_setup(a, ua, ra);
        wide_offer_setup(c, uc, rc);
        a.set_tracking(1, true); c.set_tracking(1, true);
        struct Cb {
            CT& ct; FeedTimeType t; int n = 0;
            void operator()(H, uint64_t, D) {
                ++n;
                g_nbbo[1] = {100200, 101100};
                ct.on_book_update(1, BookEvent::Execute, Side::Sell, 101100, t, 0, nop);
            }
        } cb{a, ua + 301'000};
        g_nbbo[1] = {100100, 101000};                                      // bid up a tick: near side only,
        a.on_book_update(1, BookEvent::Execute, Side::Sell, 101000, ua + 300'000, 0, cb);   // so (c) fires
        g_nbbo[1] = {100100, 101000};
        c.on_book_update(1, BookEvent::Execute, Side::Sell, 101000, uc + 300'000, 0, nop);
        g_nbbo[1] = {100200, 101100};
        c.on_book_update(1, BookEvent::Execute, Side::Sell, 101100, uc + 301'000, 0, nop);
        CHECK(cb.n == 1, "callbacks=%d", cb.n);
        CHECK(a.sigma_rate(1) == c.sigma_rate(1), "sigma nested %.9g vs sequential %.9g",
              a.sigma_rate(1), c.sigma_rate(1));
        CHECK(a.trade_gap_ns(1) == c.trade_gap_ns(1), "tau nested %.9g vs sequential %.9g",
              a.trade_gap_ns(1), c.trade_gap_ns(1));
        CHECK(a.threshold(1) == c.threshold(1) && a.sym_time(1) == c.sym_time(1), "threshold/clock differ");
    }

    // A rearmed (d) cancel needs one more counted execution to re-fire — not every one.
    static void rearm_after_outcompeted_needs_one_more_execution() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        FeedTimeType t = 1'000'000'000;
        g_nbbo[1] = {100000, 100100};
        const H h = ct.on_place(1, 1, Side::Buy, 99700, t, NQ, nop);
        int fired = 0;
        auto cb = [&](H hh, uint64_t, D d) { ++fired; CHECK(d == D::CancelOutcompeted, "d=%d", (int)d); ct.rearm(hh); };
        for (int k = 0; k < 10; ++k) { t += 1'000; exec(ct, 1, Side::Buy, 100000, t, cb); }
        CHECK(fired == 1, "fired %d after 10 executions", fired);
        CHECK(ct.order(h)->n_aggr_prints == 9 && ct.order(h)->active, "n=%u active=%d",
              ct.order(h)->n_aggr_prints, (int)ct.order(h)->active);
        t += 1'000; exec(ct, 1, Side::Sell, 100100, t, cb);                // offer lifted: no count
        CHECK(fired == 1, "fired on a non-counting execution");
        for (int k = 0; k < 5; ++k) { t += 1'000; exec(ct, 1, Side::Buy, 100000, t, cb); }
        CHECK(fired == 6, "fired %d: expected one re-fire per counted execution after rearm", fired);
        // (c) rearm is untouched: fires again on the next update while deep
        CT c2(4, 16, g_nopool); FeedTimeType u; Rec r;
        const H h2 = wide_offer_setup(c2, u, r);
        int fa = 0; auto cba = [&](H hh, uint64_t, D) { ++fa; c2.rearm(hh); };
        touch(c2, 1, u + 300'000, cba); touch(c2, 1, u + 400'000, cba);
        CHECK(fa == 2 && c2.order(h2)->active, "(c) rearm: fired %d", fa);
    }

    // ---- 5. stale handles -----------------------------------------------------------

    static void too_late_to_cancel_race_keeps_new_order() {
        MiniBook bk;
        CT ct(4, 16, bk.pool); ct.set_symbol(1, T, T);
        FeedTimeType t = 1'000'000'000;
        g_nbbo[1] = {100000, 100100};
        const H h20 = place(ct, bk, 20, 1, Side::Buy, 99500, 100, t);
        const uint32_t q20 = bk.qh(20);
        Rec r;
        for (int k = 0; k < 10; ++k) { t += 1'000; exec(ct, 1, Side::Buy, 100000, t, r.cb()); }
        CHECK(r.v.size() == 1 && r.v[0].h == h20, "setup: cancel not emitted");
        t += 1'000;                                                        // filled before the cancel landed:
        CHECK(bk.remove(20) & kQTracked, "book did not report our order");             //   the book says it is ours
        CHECK(ct.on_book_remove(1, Side::Buy, q20), "on_book_remove found no order");
        exec(ct, 1, Side::Buy, 99500, t, nop);
        const H h21 = place(ct, bk, 21, 1, Side::Buy, 99900, 100, t);      // reuses the lane
        CHECK((h21 & 0xFFFFFFFFu) == (h20 & 0xFFFFFFFFu) && h21 != h20, "setup: lane not reused with new generation");
        CHECK(!ct.on_done(h20),            "stale on_done accepted");      // "too late to cancel" for 20
        CHECK(!ct.rearm(h20),              "stale rearm accepted");
        CHECK(!ct.order(h20),              "stale handle resolves");
        CHECK(ct.evaluate(h20) == D::Hold, "stale handle evaluates");
        const auto o = ct.order(h21);
        CHECK(o && o->id == 21 && o->active && o->sz == 100 && o->px == 99900, "new order disturbed");
        CHECK(ct.find(1, Side::Buy, bk.qh(21)) == h21, "find by queue handle");
    }

    static void pool_exhaustion_and_reuse() {
        CT ct(4, 2, g_nopool); ct.set_symbol(1, T, T);
        const H a = ct.on_place(1, 1, Side::Buy, 99000, 1, NQ, nop);
        const H b = ct.on_place(2, 1, Side::Buy, 99000, 1, NQ, nop);
        CHECK(a != CT::kInvalid && b != CT::kInvalid, "placement failed");
        CHECK(ct.on_place(3, 1, Side::Buy, 99000, 1, NQ, nop) == CT::kInvalid && ct.place_failures() == 1,
              "exhaustion not reported");
        CHECK(ct.on_done(a) && ct.on_place(4, 1, Side::Buy, 99000, 1, NQ, nop) != CT::kInvalid, "freed slot not reusable");
        CHECK(!ct.on_done(CT::kInvalid), "kInvalid accepted");
    }

    // ---- 6. estimators ------------------------------------------------------------------

    static void unchanged_nbbo_does_no_estimator_work() {
        CT a(4, 16, g_nopool), c(4, 16, g_nopool);
        a.set_symbol(1, T, T); c.set_symbol(1, T, T); a.set_tracking(1, true); c.set_tracking(1, true);
        FeedTimeType t = 1'000'000'000;
        quote(a, 1, 100000, 100100, t, nop);
        quote(c, 1, 100000, 100100, t, nop);
        for (int k = 1; k <= 1000; ++k) touch(a, 1, t + FeedTimeType(k) * 1'000'000, nop);   // same pair
        touch(c, 1, t + 1000 * 1'000'000ull, nop);                                            // one update
        CHECK(a.sigma_rate(1) == c.sigma_rate(1) && a.threshold(1) == c.threshold(1) &&
              a.trade_gap_ns(1) == c.trade_gap_ns(1), "1000 same-pair updates differ from one");
    }

    static void trade_gap_seeded_with_first_gap() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T); ct.set_tracking(1, true);
        FeedTimeType t = 1'000'000'000;
        for (int k = 0; k < 6; ++k) { t += 10'000'000'000ull; exec(ct, 1, Side::Buy, 100000, t, nop); }
        CHECK(std::fabs(ct.trade_gap_ns(1) / 1e10 - 1.0) < 1e-9, "gap EWMA %.4g, true 1e10", ct.trade_gap_ns(1));
    }

    static void same_timestamp_executions_are_one_event() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T); ct.set_tracking(1, true);
        FeedTimeType t = 1'000'000'000;
        for (int k = 0; k < 100; ++k) {
            t += 1'000'000;
            for (int j = 0; j < 5; ++j) exec(ct, 1, Side::Buy, 100000 - j * T, t, nop);   // one sweep, 5 fills
        }
        CHECK(std::fabs(ct.trade_gap_ns(1) / 1e6 - 1.0) < 1e-9, "gap EWMA %.4g, true 1e6", ct.trade_gap_ns(1));
    }

    static void same_timestamp_mid_burst_records_net_move() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T); ct.set_tracking(1, true);
        FeedTimeType t = 1'000'000'000;
        quote(ct, 1, 100000, 100100, t, nop);
        t += 1'000'000'000;
        quote(ct, 1, 100100, 100200, t, nop);                              // three 1-tick steps
        quote(ct, 1, 100200, 100300, t, nop);                              // in one timestamp
        quote(ct, 1, 100300, 100400, t, nop);
        const double s1 = ct.sigma_rate(1), v1 = s1 * s1 * 1e9 / (T * T); // tick^2 per second
        CHECK(std::fabs(v1 - 9.0) < 1e-6, "3-tick move in one ts: %.4f tick^2/s, expected 9", v1);

        CT c2(4, 16, g_nopool); c2.set_symbol(1, T, T); c2.set_tracking(1, true);
        FeedTimeType u = 1'000'000'000;
        quote(c2, 1, 100000, 100100, u, nop);
        u += 1'000'000'000;
        quote(c2, 1, 100000, 100300, u, nop);                              // offer flickers out
        quote(c2, 1, 100000, 100100, u, nop);                              // and back, same ts
        CHECK(c2.sigma_rate(1) == 0.0, "intra-timestamp flicker added variance: %.4g", c2.sigma_rate(1));
    }

    // ---- 7. fixed historical spread ----------------------------------------------------

    static void hist_spread_floor_is_fixed_per_symbol() {
        CancelParams p; p.vol_horizon_cap_ns = 1;                           // no vol floor
        CT ct(4, 16, g_nopool, p);
        ct.set_symbol(1, T, 600);                                           // 6 ticks -> floor 9 ticks
        ct.set_tracking(1, true);
        FeedTimeType t = 1'000'000'000;
        quote(ct, 1, 100000, 100100, t, nop);
        CHECK(ct.threshold(1) == 900, "threshold %lld, expected 1.5 x 6 ticks", (long long)ct.threshold(1));
        for (int k = 0; k < 600; ++k) { t += 100'000'000; quote(ct, 1, 100000, k & 1 ? 101400 : 101500, t, nop); }   // ends at 14 ticks
        CHECK(ct.threshold(1) == 1400, "wide spread: threshold %lld, expected the spread", (long long)ct.threshold(1));
        t += 100'000'000; quote(ct, 1, 100000, 100100, t, nop);            // a minute of 14-15 tick spreads
        CHECK(ct.threshold(1) == 900, "threshold %lld after wide spreads: the floor has no memory",
              (long long)ct.threshold(1));
        Rec r;
        const H h = ct.on_place(1, 1, Side::Buy, 99500, t, NQ, nop);            // depth 6 ticks
        t += 1'000'000; quote(ct, 1, 100300, 100400, t, r.cb());            // depth 9 ticks: at the floor
        touch(ct, 1, t + 300'000, r.cb());
        CHECK(r.v.empty(), "fired at depth == 1.5 x hist spread");
        t += 1'000'000; quote(ct, 1, 100400, 100500, t, r.cb());            // depth 10 ticks
        touch(ct, 1, t + 300'000, r.cb());
        CHECK(r.v.size() == 1 && r.v[0].h == h && r.v[0].d == D::CancelDepth, "n=%zu", r.v.size());
        CancelParams q = p; q.hist_spread_mult = 2.0;                      // params apply to the stored spread
        ct.set_params(q);
        CHECK(ct.threshold(1) == 1200, "threshold %lld after hist_spread_mult 2", (long long)ct.threshold(1));
        q.min_ticks = 20; ct.set_params(q);
        CHECK(ct.threshold(1) == 2000, "threshold %lld with a 20-tick floor", (long long)ct.threshold(1));
    }

    // ---- 8. tracking gate ----------------------------------------------------------------

    static void untracked_symbol_costs_one_bit_test() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        const uint64_t reads = g_nbbo_reads;
        for (int k = 1; k <= 100; ++k) {
            touch(ct, 1, FeedTimeType(k) * 1'000'000, nop);
            g_nbbo[1] = {100000 + (k & 1) * T, 100100 + (k & 1) * T};
            exec(ct, 1, Side::Buy, 100000, FeedTimeType(k) * 1'000'000 + 1, nop);
        }
        CHECK(!ct.tracking(1), "symbol with no orders is tracked");
        CHECK(g_nbbo_reads == reads, "untracked updates read the NBBO %llu times",
              (unsigned long long)(g_nbbo_reads - reads));
        CHECK(ct.sym_time(1) == 0 && ct.sigma_rate(1) == 0.0 && ct.trade_gap_ns(1) == 0.0,
              "untracked updates changed state");
    }

    static void tracking_follows_active_orders() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        FeedTimeType t = 1'000'000'000;
        g_nbbo[1] = {100000, 100100};
        const H h = ct.on_place(1, 1, Side::Buy, 99700, t, NQ, nop);
        CHECK(ct.tracking(1), "placed order not tracked");
        for (int k = 0; k < 10; ++k) { t += 1'000; exec(ct, 1, Side::Buy, 100000, t, nop); }   // (d) decides
        CHECK(!ct.tracking(1), "still tracked with no active order (cancel pending)");
        const FeedTimeType frozen = ct.sym_time(1);
        exec(ct, 1, Side::Buy, 100000, t + 5'000, nop);
        CHECK(ct.sym_time(1) == frozen, "untracked update advanced the clock");
        CHECK(ct.rearm(h) && ct.tracking(1), "rearm did not resume tracking");
        exec(ct, 1, Side::Buy, 100000, t + 6'000, nop);
        CHECK(ct.sym_time(1) == t + 6'000, "rearmed symbol not updated");
        CHECK(ct.on_done(h) && !ct.tracking(1), "released last order, still tracked");
    }

    static void pin_keeps_estimators_warm() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        ct.set_tracking(1, true);
        CHECK(ct.tracking(1), "pin ignored");
        FeedTimeType t = 1'000'000'000;
        PriceType b = 100000;
        for (int k = 0; k < 100; ++k) { t += 10'000'000; b += (k & 1) ? T : -T; quote(ct, 1, b, b + T, t, nop); }
        CHECK(ct.sigma_rate(1) > 0.0 && ct.sym_time(1) == t, "pinned symbol not updated");
        const H h = ct.on_place(1, 1, Side::Buy, 99000, t, NQ, nop);
        ct.set_tracking(1, false);
        CHECK(ct.tracking(1), "unpin stopped tracking a symbol with an active order");
        ct.on_done(h);
        CHECK(!ct.tracking(1), "unpinned, no orders, still tracked");
        const double sig = ct.sigma_rate(1);
        quote(ct, 1, b + 5 * T, b + 6 * T, t + 10'000'000, nop);
        CHECK(ct.sigma_rate(1) == sig && ct.sym_time(1) == t, "untracked symbol updated");
    }

    // Executions are not seen while untracked: when tracking resumes, the next
    // execution starts a new gap instead of closing one across the stretch.
    static void resume_restarts_the_trade_gap() {
        CT ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        ct.set_tracking(1, true);
        FeedTimeType t = 1'000'000'000;
        for (int k = 0; k < 20; ++k) { t += 1'000'000; exec(ct, 1, Side::Buy, 100000, t, nop); }
        CHECK(ct.trade_gap_ns(1) == 1e6, "setup: gap EWMA %.6g", ct.trade_gap_ns(1));
        ct.set_tracking(1, false);
        for (int k = 0; k < 1000; ++k) { t += 10'000'000; exec(ct, 1, Side::Buy, 100000, t, nop); }   // unseen
        ct.set_tracking(1, true);
        t += 1'000'000; exec(ct, 1, Side::Buy, 100000, t, nop);
        t += 1'000'000; exec(ct, 1, Side::Buy, 100000, t, nop);
        CHECK(ct.trade_gap_ns(1) == 1e6, "gap EWMA %.6g after resuming: the 10 s stretch leaked in",
              ct.trade_gap_ns(1));
    }

    // ---- 9. orders in the book (QBlockList) ------------------------------------------------

    static void on_place_flags_the_queue_slot() {
        MiniBook bk;
        CT ct(4, 16, bk.pool); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        bk.add(1000, 1, Side::Buy, 99900, 100);                            // venue orders
        bk.add(1001, 1, Side::Buy, 99900, 200);
        const H h = place(ct, bk, 7, 1, Side::Buy, 99900, 300, 1'000);
        bk.add(1002, 1, Side::Buy, 99900, 400);                            // behind us
        CHECK(QBlockList::tracked(bk.pool, bk.qh(7)), "our slot not flagged");
        CHECK(!QBlockList::tracked(bk.pool, bk.qh(1000)) && !QBlockList::tracked(bk.pool, bk.qh(1002)),
              "venue orders flagged");
        const auto o = ct.order(h);
        CHECK(o && o->queue_ahead == 2 && o->queue_ahead_sz == 300 && o->sz == 300 && o->qh == bk.qh(7),
              "position %u/%llu size %u", o->queue_ahead, (unsigned long long)o->queue_ahead_sz, o->sz);
        CHECK(!(bk.reduce(1000, 40) & kQTracked) && (bk.reduce(7, 100) & kQTracked), "reduce did not report the flag");
        const auto o2 = ct.order(h);
        CHECK(o2->sz == 200 && o2->queue_ahead_sz == 260, "after partials: size %u ahead %llu",
              o2->sz, (unsigned long long)o2->queue_ahead_sz);
        CHECK(!(bk.remove(1001) & kQTracked), "venue order reported tracked");
        CHECK(ct.order(h)->queue_ahead == 1 && ct.order(h)->queue_ahead_sz == 60, "ahead after a cancel ahead");
    }

    // Our order leaves the book: remove() reports it, on_book_remove releases
    // the lane, and the symbol stops being tracked with its last order.
    static void book_remove_releases_the_lane() {
        MiniBook bk;
        CT ct(4, 16, bk.pool); ct.set_symbol(1, T, T);
        FeedTimeType t = 1'000'000'000;
        g_nbbo[1] = {100000, 100100};
        const H a = place(ct, bk, 7, 1, Side::Buy, 99900, 100, t);
        const H b = place(ct, bk, 8, 1, Side::Buy, 99800, 100, t);
        const uint32_t qa = bk.qh(7);
        CHECK(ct.find(1, Side::Buy, qa) == a && ct.find(1, Side::Sell, qa) == CT::kInvalid, "find");
        CHECK(bk.remove(7) & kQTracked, "our order not reported by remove()");
        CHECK(QBlockList::tracked(bk.pool, bk.qh(8)), "sibling flag lost");
        CHECK(ct.on_book_remove(1, Side::Buy, qa) && !ct.order(a) && ct.live_orders() == 1, "lane not released");
        exec(ct, 1, Side::Buy, 99900, t + 1'000, nop);                     // the execution that filled it
        CHECK(ct.order(b)->n_aggr_prints == 1, "sibling did not count the execution");
        const uint32_t qb = bk.qh(8);
        CHECK((bk.remove(8) & kQTracked) && ct.on_book_remove(1, Side::Buy, qb), "second remove");
        CHECK(ct.live_orders() == 0 && !ct.tracking(1), "symbol tracked with no orders left");
    }

    // on_done while the order is still in the book (cancel ack first): the
    // flag stays with the order until the book removes it, and the late
    // on_book_remove is a no-op.
    static void flag_after_on_done_is_harmless() {
        MiniBook bk;
        CT ct(4, 16, bk.pool); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        const H a = place(ct, bk, 7, 1, Side::Buy, 99900, 100, 1'000);
        const H b = place(ct, bk, 8, 1, Side::Buy, 99900, 100, 1'000);
        const uint32_t qa = bk.qh(7);
        CHECK(ct.on_done(a), "on_done");
        CHECK(bk.remove(7) & kQTracked, "flag cleared before the book removed the order");
        CHECK(!ct.on_book_remove(1, Side::Buy, qa), "late on_book_remove released something");
        CHECK(ct.order(b) && ct.live_orders() == 1, "sibling disturbed");
        const uint32_t qn = bk.add(1000, 1, Side::Buy, 99900, 50);         // a venue order reuses the slot
        CHECK(qn == qa && !QBlockList::tracked(bk.pool, qn), "reused slot starts tracked");
    }

    // A lane whose order left the book without on_book_remove (a host that
    // missed the report) reads as "not in the book" — never as the position
    // or size of whatever order reuses the slot — and its queue gate passes.
    static void missed_remove_never_reads_another_order() {
        MiniBook bk;
        CancelParams p; p.aggr_gate_n = 3;
        CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
        FeedTimeType t = 1'000'000'000;
        g_nbbo[1] = {100000, 100100};
        bk.add(1000, 1, Side::Buy, 99700, 100);
        const H h = place(ct, bk, 7, 1, Side::Buy, 99700, 100, t);
        const uint32_t q7 = bk.qh(7);
        bk.remove(7);                                                     // report ignored
        CHECK(bk.add(1001, 1, Side::Buy, 99700, 555) == q7, "setup: slot not reused");
        bk.add(1002, 1, Side::Buy, 99700, 1);
        const auto o = ct.order(h);
        CHECK(o && o->sz == 0 && o->queue_ahead == 0 && o->queue_ahead_sz == 0,
              "read the slot's new order: size %u, ahead %u", o->sz, o->queue_ahead);
        int fired = 0;
        auto cb = [&](H, uint64_t, D) { ++fired; };
        for (int k = 0; k < 10; ++k) { t += 1'000; exec(ct, 1, Side::Buy, 100000, t, cb); }
        CHECK(fired == 1, "gate used another order's position (fired %d)", fired);
    }

    // (d)'s queue gate reads our position from the book.
    static void queue_gate_reads_position_from_book() {
        auto scenario = [](uint32_t min_n, uint64_t min_sz, bool remove_ahead) {
            MiniBook bk;
            CancelParams p; p.aggr_gate_n = min_n; p.aggr_gate_sz = min_sz;
            CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
            FeedTimeType t = 1'000'000'000;
            g_nbbo[1] = {100000, 100100};
            bk.add(1000, 1, Side::Buy, 99700, 100);
            bk.add(1001, 1, Side::Buy, 99700, 200);
            bk.add(1002, 1, Side::Buy, 99700, 300);                       // 3 orders, 600 shares ahead
            place(ct, bk, 7, 1, Side::Buy, 99700, 100, t);
            bk.add(1003, 1, Side::Buy, 99700, 900);                       // behind us: no effect
            if (remove_ahead) bk.remove(1001);
            int fired = 0;
            auto cb = [&](H, uint64_t, D) { ++fired; };
            for (int k = 0; k < 10; ++k) { t += 1'000; exec(ct, 1, Side::Buy, 100000, t, cb); }
            return fired;
        };
        CHECK(scenario(3, 0, false) == 1,   "3 ahead, gate 3: did not fire");
        CHECK(scenario(4, 0, false) == 0,   "3 ahead, gate 4: fired");
        CHECK(scenario(0, 600, false) == 1, "600 shares ahead, gate 600: did not fire");
        CHECK(scenario(0, 601, false) == 0, "600 shares ahead, gate 601: fired");
        CHECK(scenario(3, 0, true) == 0,    "cancel ahead left 2 ahead, gate 3: fired");
        CHECK(scenario(2, 400, true) == 1,  "2 ahead / 400 shares, gate 2 / 400: did not fire");
        {   // gate blocks, then the queue changes and the next counted execution re-checks it
            MiniBook bk;
            CancelParams p; p.aggr_gate_n = 2;
            CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
            FeedTimeType t = 1'000'000'000;
            g_nbbo[1] = {100000, 100100};
            bk.add(1000, 1, Side::Buy, 99700, 100);
            place(ct, bk, 7, 1, Side::Buy, 99700, 100, t);                 // 1 ahead: gate closed
            int fired = 0;
            auto cb = [&](H, uint64_t, D) { ++fired; };
            for (int k = 0; k < 12; ++k) { t += 1'000; exec(ct, 1, Side::Buy, 100000, t, cb); }
            CHECK(fired == 0, "fired with 1 ahead, gate 2");
            CHECK(QBlockList::ahead_of(bk.pool, bk.qh(7)).n == 1, "setup");
            CancelParams q = p; q.aggr_gate_n = 1; ct.set_params(q);
            t += 1'000; exec(ct, 1, Side::Buy, 100000, t, cb);
            CHECK(fired == 1, "gate re-check on the next counted execution: fired %d", fired);
        }
        {   // position unknown (not in the book's queues): the gate passes
            CancelParams p; p.aggr_gate_n = 5;
            CT ct(4, 16, g_nopool, p); ct.set_symbol(1, T, T);
            FeedTimeType t = 1'000'000'000;
            g_nbbo[1] = {100000, 100100};
            ct.on_place(7, 1, Side::Buy, 99700, t, NQ, nop);
            int fired = 0;
            auto cb = [&](H, uint64_t, D) { ++fired; };
            for (int k = 0; k < 10; ++k) { t += 1'000; exec(ct, 1, Side::Buy, 100000, t, cb); }
            CHECK(fired == 1, "unknown position blocked by the gate");
        }
    }

    // A level past 16 orders spills into a chain; ahead_of walks it backwards.
    static void spilled_level_queue_position() {
        MiniBook bk;
        CT ct(4, 16, bk.pool); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        for (int k = 0; k < 37; ++k) bk.add(1000 + k, 1, Side::Buy, 99900, 10 + k);
        const H h = place(ct, bk, 7, 1, Side::Buy, 99900, 50, 1'000);      // 38th: third block
        for (int k = 0; k < 37; k += 3) bk.remove(1000 + k);               // holes in every block
        uint32_t n = 0; uint64_t sz = 0;
        for (int k = 0; k < 37; ++k) if (k % 3) { ++n; sz += 10 + k; }
        const QAhead a = bk.level(1, Side::Buy, 99900).ahead(bk.pool, bk.qh(7));
        const auto o = ct.order(h);
        CHECK(o->queue_ahead == n && o->queue_ahead_sz == sz && a.n == n && a.sz == sz,
              "ahead %u/%llu, list %u/%llu, expected %u/%llu", o->queue_ahead,
              (unsigned long long)o->queue_ahead_sz, a.n, (unsigned long long)a.sz, n, (unsigned long long)sz);
    }

    // ---- 10. (a) queue position at placement -----------------------------------------------

    // Five orders (1000 shares) ahead of our 100: N = 3, K = 2 -> (a) at once,
    // from inside on_place; the order is decided, the symbol untracked.
    static void queue_fires_at_placement() {
        MiniBook bk;
        CancelParams p; p.queue_n = 3; p.queue_k = 2.0;
        CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        for (uint64_t r = 1000; r < 1005; ++r) bk.add(r, 1, Side::Buy, 99900, 200);
        Rec rec; int during = -1;
        const H h = place_cb(ct, bk, 7, 1, Side::Buy, 99900, 100, 1'000,
                             [&](H x, uint64_t id, D d) { during = int(rec.v.size()); rec.v.push_back({x, id, d}); });
        CHECK(rec.v.size() == 1 && during == 0 && rec.v[0].d == D::CancelQueue && rec.v[0].h == h && rec.v[0].id == 7,
              "decisions %zu", rec.v.size());
        CHECK(ct.order(h) && !ct.order(h)->active && !ct.tracking(1), "order still active / symbol tracked");
        CHECK(ct.evaluate(h) == D::CancelQueue, "evaluate disagrees");
    }

    // Both limits must hold; N = 0 is off; K scales with our size; an order
    // not in the book's queues is never judged.
    static void queue_needs_both_limits() {
        struct Case { uint32_t n; double k; uint32_t ours; bool fire; const char* what; };
        const Case cs[] = {
            {3, 2.0, 100, true,  "5 orders / 1000 shares vs 3 / 200"},
            {5, 2.0, 100, true,  "exactly N = 5 orders"},
            {6, 2.0, 100, false, "too few orders"},
            {3, 6.0, 200, false, "too few shares (6 x 200 = 1200)"},
            {3, 5.0, 200, true,  "exactly 5 x 200 = 1000"},
            {0, 0.0, 100, false, "queue_n = 0: off"},
        };
        for (const Case& c : cs) {
            MiniBook bk;
            CancelParams p; p.queue_n = c.n; p.queue_k = c.k;
            CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
            g_nbbo[1] = {100000, 100100};
            for (uint64_t r = 1000; r < 1005; ++r) bk.add(r, 1, Side::Buy, 99900, 200);
            Rec rec;
            place_cb(ct, bk, 7, 1, Side::Buy, 99900, c.ours, 1'000, rec.cb());
            CHECK((rec.v.size() == 1) == c.fire, "%s: %zu decisions", c.what, rec.v.size());
        }
        CancelParams p; p.queue_n = 1;
        CT ct(4, 16, g_nopool, p); ct.set_symbol(1, T, T);
        Rec rec;
        ct.on_place(7, 1, Side::Buy, 99900, 1'000, NQ, rec.cb());
        CHECK(rec.v.empty(), "judged an order outside the queues");
    }

    // Defaults: N = 5, K = 3. Five orders of 100 ahead of our 100 (500 >=
    // 300) fire; four do not.
    static void queue_defaults_five_and_three() {
        for (int n = 4; n <= 5; ++n) {
            MiniBook bk;
            CT ct(4, 16, bk.pool); ct.set_symbol(1, T, T);
            g_nbbo[1] = {100000, 100100};
            for (int k = 0; k < n; ++k) bk.add(1000 + uint64_t(k), 1, Side::Buy, 99900, 100);
            Rec r;
            place_cb(ct, bk, 7, 1, Side::Buy, 99900, 100, 1'000, r.cb());
            CHECK((r.v.size() == 1 && r.v[0].d == D::CancelQueue) == (n == 5), "%d ahead: %zu decisions", n,
                  r.v.size());
        }
        const CancelParams d;
        CHECK(d.queue_n == 5 && d.queue_k == 3.0 && d.abandon_pct == 50.0, "defaults %u %g %g", d.queue_n,
              d.queue_k, d.abandon_pct);
    }

    // (a) after placement: our order reduced lowers K x our size. Three
    // orders of 200 ahead of our 300 (600 < 900) hold; one of them cut by 50
    // re-checks nothing; ours cut to 150 (550 >= 450) decides (a) on the
    // update that reports it (kQTracked).
    static void queue_rechecked_when_ours_is_reduced() {
        MiniBook bk;
        CancelParams p; p.queue_n = 3; p.queue_k = 3.0; p.abandon_pct = -1.0;
        CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        for (uint64_t k = 1000; k < 1003; ++k) bk.add(k, 1, Side::Buy, 99900, 200);
        Rec r;
        const H h = place_cb(ct, bk, 7, 1, Side::Buy, 99900, 300, 1'000, r.cb());
        CHECK(r.v.empty(), "fired at placement");
        const uint8_t f1 = bk.reduce(1000, 50);
        ct.on_book_update(1, BookEvent::Cancel, Side::Buy, 99900, 2'000, f1, r.cb());
        CHECK(r.v.empty() && f1 == 0, "fired on a reduction ahead (flags %u)", f1);
        const uint8_t f2 = bk.reduce(7, 150);
        ct.on_book_update(1, BookEvent::Cancel, Side::Buy, 99900, 3'000, f2, r.cb());
        CHECK(f2 == kQTracked && r.v.size() == 1 && r.v[0].h == h && r.v[0].d == D::CancelQueue &&
              !ct.order(h)->active && !ct.tracking(1), "flags %u, %zu decisions", f2, r.v.size());
    }

    // Two of ours decided by (a) in one pass (the shares ahead of both grew:
    // re-checked on that update with kSizeUp, else on the next reduction of
    // ours); the first callback frees both, so the second decision drops.
    static void queue_decision_dropped_when_freed_first() {
        MiniBook bk;
        CancelParams p; p.queue_n = 3; p.queue_k = 5.0; p.abandon_pct = -1.0;
        CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        for (uint64_t k = 1000; k < 1003; ++k) bk.add(k, 1, Side::Buy, 99900, 100);
        const H x = place(ct, bk, 7, 1, Side::Buy, 99900, 100, 1'000);      // 300 ahead < 500
        const H y = place(ct, bk, 8, 1, Side::Buy, 99900, 100, 1'000);      // 400 ahead < 500
        int calls = 0;
        auto cb = [&](H h, uint64_t, D d) {
            ++calls;
            CHECK(h == x && d == D::CancelQueue, "first decision %d", int(d));
            ct.on_done(x); ct.on_done(y);
        };
        uint8_t f = bk.grow(1000, 300);                                     // 600 / 700 ahead >= 500
        ct.on_book_update(1, BookEvent::Modify, Side::Buy, 99900, 2'000, f, cb);
        if (calls == 0) {
            f = bk.reduce(8, 1);
            ct.on_book_update(1, BookEvent::Cancel, Side::Buy, 99900, 3'000, f, cb);
        }
        CHECK(calls == 1 && !ct.order(x) && !ct.order(y), "calls=%d: a freed order's (a) delivered", calls);
    }

    // ---- 11. (b) at the front after the round lots ahead cancelled ---------------------------

    // Ahead: four round lots and an odd lot; behind: an order that joins
    // after ours and stays. Three round lots cancel and one fills: 3/4 = 75%
    // > 50% (the default X), but the odd lot is still ahead. (b) fires on the
    // removal that puts ours at the front — the odd lot's, unflagged — not
    // before, and not waiting for the order behind.
    static void abandon_fires_at_the_front() {
        MiniBook bk;
        CancelParams p; p.queue_n = 0;                                        // (b) alone
        CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        FeedTimeType t = 1'000'000;
        for (uint64_t k = 1000; k < 1004; ++k) bk.add(k, 1, Side::Buy, 99900, 200);
        bk.add(1004, 1, Side::Buy, 99900, 50);                                // odd lot
        const H h = place(ct, bk, 7, 1, Side::Buy, 99900, 100, t);
        bk.add(1005, 1, Side::Buy, 99900, 300);                               // behind ours, stays
        CHECK(ct.order(h)->ahead_lot_n == 4, "counted %u round lots ahead", ct.order(h)->ahead_lot_n);
        Rec r;
        CHECK(leave(ct, bk, 1000, BookEvent::Delete, t += 1000, r.cb()) == (kQGone | kQAhead), "flags of a counted order");
        leave(ct, bk, 1001, BookEvent::Execute, t += 1000, r.cb());          // filled: not a cancel
        leave(ct, bk, 1002, BookEvent::Cancel, t += 1000, r.cb());           // cancelled to zero
        leave(ct, bk, 1003, BookEvent::Replace, t += 1000, r.cb());          // replaced away
        CHECK(r.v.empty() && ct.order(h)->ahead_cancelled == 3 && ct.evaluate(h) == D::Hold,
              "early decision / %u cancelled / evaluate %d", ct.order(h)->ahead_cancelled, int(ct.evaluate(h)));
        CHECK(leave(ct, bk, 1004, BookEvent::Delete, t += 1000, r.cb()) == kQGone, "odd lot flagged");
        CHECK(r.v.size() == 1 && r.v[0].d == D::CancelAbandoned && r.v[0].h == h, "decisions %zu", r.v.size());
        CHECK(ct.order(h)->ahead_cancelled == 3 && !ct.order(h)->active && !QBlockList::alone(bk.pool, bk.qh(7)),
              "state after (b)");
        CHECK(!ct.tracking(1), "symbol still tracked with its only order decided");
    }

    // 2 of 4 cancelled is 50%, not more: no (b) — and none later: at the
    // front, nothing ahead can leave any more.
    static void abandon_needs_more_than_x() {
        MiniBook bk;
        CancelParams p; p.queue_n = 0;
        CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        FeedTimeType t = 1'000'000;
        for (uint64_t k = 1000; k < 1004; ++k) bk.add(k, 1, Side::Buy, 99900, 200);
        const H h = place(ct, bk, 7, 1, Side::Buy, 99900, 100, t);
        Rec r;
        leave(ct, bk, 1000, BookEvent::Delete, t += 1000, r.cb());
        leave(ct, bk, 1001, BookEvent::Delete, t += 1000, r.cb());
        leave(ct, bk, 1002, BookEvent::Execute, t += 1000, r.cb());
        leave(ct, bk, 1003, BookEvent::Execute, t += 1000, r.cb());
        bk.add(1004, 1, Side::Buy, 99900, 300);                               // behind, then gone
        leave(ct, bk, 1004, BookEvent::Delete, t += 1000, r.cb());
        CHECK(r.v.empty() && ct.order(h)->active && QBlockList::ahead_n(bk.pool, bk.qh(7)) == 0, "fired at 50%%");
    }

    // What is behind ours never matters: an order behind leaving is not
    // counted and fires nothing. A partial cancel ahead never counts; the
    // same order cancelled to zero does.
    static void abandon_ignores_what_is_behind() {
        MiniBook bk;
        CancelParams p; p.queue_n = 0;
        CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        FeedTimeType t = 1'000'000;
        bk.add(1000, 1, Side::Buy, 99900, 300);
        bk.add(1001, 1, Side::Buy, 99900, 300);
        const H h = place(ct, bk, 7, 1, Side::Buy, 99900, 100, t);
        bk.add(1002, 1, Side::Buy, 99900, 500);                               // behind us
        Rec r;
        const uint8_t pf = bk.reduce(1000, 250);                              // partial cancel: 50 left
        ct.on_book_update(1, BookEvent::Cancel, Side::Buy, 99900, t += 1000, pf, r.cb());
        CHECK(pf == kQAhead && ct.order(h)->ahead_cancelled == 0, "partial counted (flags %u)", pf);
        CHECK(leave(ct, bk, 1002, BookEvent::Delete, t += 1000, r.cb()) == kQGone && r.v.empty() &&
              ct.order(h)->ahead_cancelled == 0, "the order behind counted or fired");
        leave(ct, bk, 1000, BookEvent::Cancel, t += 1000, r.cb());           // now to zero: counts
        CHECK(r.v.empty() && ct.order(h)->ahead_cancelled == 1, "1 of 2: %u", ct.order(h)->ahead_cancelled);
        leave(ct, bk, 1001, BookEvent::Delete, t += 1000, r.cb());           // at the front, 2 of 2
        CHECK(r.v.size() == 1 && r.v[0].d == D::CancelAbandoned && ct.order(h)->ahead_cancelled == 2,
              "decisions %zu", r.v.size());
    }

    // Every order of ours has its own (b), judged in queue order. Ahead of X:
    // R1 R2; between X and Y: R3; behind Y: B. X counts R1 R2; Y counts R1
    // R2 R3 — not X, which is ours. R1 and R2 cancel: X at the front, 2/2,
    // (b) for X; Y is not at the front while X is there. The host cancels X:
    // its departure does not count for Y. R3 fills: Y at the front with 2 of
    // 3 cancelled, (b) for Y. With (b) off nothing is flagged.
    static void abandon_each_order_in_queue_order() {
        MiniBook bk;
        CancelParams p; p.queue_n = 0;
        CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        FeedTimeType t = 1'000'000;
        bk.add(1000, 1, Side::Buy, 99900, 200);
        bk.add(1001, 1, Side::Buy, 99900, 200);
        const H x = place(ct, bk, 7, 1, Side::Buy, 99900, 100, t);
        bk.add(1002, 1, Side::Buy, 99900, 200);
        const H y = place(ct, bk, 8, 1, Side::Buy, 99900, 100, t);
        bk.add(1003, 1, Side::Buy, 99900, 300);
        CHECK(ct.order(x)->ahead_lot_n == 2 && ct.order(y)->ahead_lot_n == 3, "n0 %u / %u",
              ct.order(x)->ahead_lot_n, ct.order(y)->ahead_lot_n);
        Rec r;
        leave(ct, bk, 1000, BookEvent::Delete, t += 1000, r.cb());
        leave(ct, bk, 1001, BookEvent::Delete, t += 1000, r.cb());
        CHECK(r.v.size() == 1 && r.v[0].h == x && r.v[0].d == D::CancelAbandoned &&
              ct.order(y)->ahead_cancelled == 2 && ct.order(y)->active, "X first: %zu decisions", r.v.size());
        CHECK(leave(ct, bk, 7, BookEvent::Delete, t += 1000, r.cb()) == (kQGone | kQTracked) && r.v.size() == 1 &&
              ct.order(y)->ahead_cancelled == 2, "X's departure counted for Y");
        leave(ct, bk, 1002, BookEvent::Execute, t += 1000, r.cb());
        CHECK(r.v.size() == 2 && r.v[1].h == y && r.v[1].d == D::CancelAbandoned, "Y: %zu decisions", r.v.size());
        MiniBook bk2;
        CancelParams q; q.abandon_pct = -1.0;
        CT c2(4, 16, bk2.pool, q); c2.set_symbol(1, T, T);
        bk2.add(1000, 1, Side::Buy, 99900, 300);
        const H h = c2.on_place(7, 1, Side::Buy, 99900, 1'000, bk2.add(7, 1, Side::Buy, 99900, 100), nop);
        CHECK(c2.order(h)->ahead_lot_n == 0 && bk2.remove(1000) == kQGone, "(b) off but flags set");
    }

    // Our orders at two prices on one side: the cancellations ahead of the
    // one at 99.80 leave the count of the one at 99.90 alone, though it is at
    // the front at 50% and one more would tip it.
    static void abandon_counts_its_own_level_only() {
        MiniBook bk;
        CancelParams p; p.queue_n = 0;
        CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        FeedTimeType t = 1'000'000;
        bk.add(1000, 1, Side::Buy, 99900, 200);
        bk.add(1001, 1, Side::Buy, 99900, 200);
        const H x = place(ct, bk, 7, 1, Side::Buy, 99900, 100, t);
        bk.add(2000, 1, Side::Buy, 99800, 200);
        bk.add(2001, 1, Side::Buy, 99800, 200);
        const H y = place(ct, bk, 8, 1, Side::Buy, 99800, 100, t);
        Rec r;
        leave(ct, bk, 1000, BookEvent::Delete, t += 1000, r.cb());          // x: 1 of 2 cancelled
        leave(ct, bk, 1001, BookEvent::Execute, t += 1000, r.cb());         // x: at the front at 50%
        CHECK(r.v.empty() && ct.order(x)->ahead_cancelled == 1 && QBlockList::ahead_n(bk.pool, bk.qh(7)) == 0, "setup");
        CHECK(leave(ct, bk, 2000, BookEvent::Delete, t += 1000, r.cb()) == (kQGone | kQAhead), "flags");
        CHECK(r.v.empty() && ct.order(x)->ahead_cancelled == 1 && ct.order(y)->ahead_cancelled == 1,
              "counted across levels: x %u, y %u (%zu decisions)", ct.order(x)->ahead_cancelled,
              ct.order(y)->ahead_cancelled, r.v.size());
        leave(ct, bk, 2001, BookEvent::Delete, t += 1000, r.cb());          // y: 2 of 2, at the front
        CHECK(r.v.size() == 1 && r.v[0].h == y && r.v[0].d == D::CancelAbandoned, "decisions %zu", r.v.size());
    }

    // A decided order is decided once. After (a) at placement, (c), (d),
    // (b) or (a) on a reduction of ours, with the host's cancel in flight (no
    // on_done), the round lots ahead have all cancelled and an order joins
    // behind and leaves — (b) holds at 0% — but it is neither decided again
    // nor counting. The symbol is pinned so every update is processed.
    static void decided_order_is_decided_once() {
        const D first[] = {D::CancelQueue, D::CancelDepth, D::CancelOutcompeted, D::CancelAbandoned, D::CancelQueue};
        for (int via = 0; via < 5; ++via) {
            MiniBook bk;
            CancelParams p; p.abandon_pct = 0.0;
            p.queue_n = via == 0 ? 1 : via == 4 ? 2 : 0;
            if (via == 4) p.queue_k = 7.0;                                    // 600 ahead < 7 x 100
            if (via == 2) p.aggr_prints_n = 3;
            CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
            ct.set_tracking(1, true);
            FeedTimeType u = 2'000'000'000;
            Rec r;
            quote(ct, 1, 100000, 100100, u, nop);
            for (int k = 0; k < 30; ++k) { u += 1'000'000; exec(ct, 1, Side::Buy, 100000, u, nop); }   // tau 1 ms
            bk.add(1000, 1, Side::Buy, 99800, 300);
            bk.add(1001, 1, Side::Buy, 99800, 300);
            const H h = place_cb(ct, bk, 2, 1, Side::Buy, 99800, 100, u, r.cb());
            if (via == 1) {                                                   // the offer gaps out, then settles
                u += 1'000'000; quote(ct, 1, 100000, 101000, u, r.cb());
                u += 300'000;   touch(ct, 1, u, r.cb());
            }
            if (via == 2) for (int k = 0; k < 3; ++k) { u += 1'000; exec(ct, 1, Side::Buy, 100000, u, r.cb()); }
            if (via == 4) {                                                   // ours cut to 80: 600 >= 560
                const uint8_t f = bk.reduce(2, 20);
                ct.on_book_update(1, BookEvent::Cancel, Side::Buy, 99800, u += 1'000, f, r.cb());
            }
            const bool b_first = via == 3;
            if (b_first) {                                                    // (b) itself: both cancel
                leave(ct, bk, 1000, BookEvent::Delete, u += 1'000, r.cb());
                leave(ct, bk, 1001, BookEvent::Delete, u += 1'000, r.cb());
            }
            CHECK(r.v.size() == 1 && r.v[0].h == h && r.v[0].d == first[via] && ct.order(h)->ahead_lot_n == 2,
                  "via %d: first decision (%zu)", via, r.v.size());
            if (!b_first) {
                leave(ct, bk, 1000, BookEvent::Delete, u += 1'000, r.cb());
                leave(ct, bk, 1001, BookEvent::Delete, u += 1'000, r.cb());
            }
            bk.add(3000, 1, Side::Buy, 99800, 100);                           // joins behind, then leaves
            ct.on_book_update(1, BookEvent::Add, Side::Buy, 99800, u += 1'000, 0, r.cb());
            leave(ct, bk, 3000, BookEvent::Delete, u += 1'000, r.cb());
            CHECK(r.v.size() == 1 && QBlockList::alone(bk.pool, bk.qh(2)) &&
                  ct.order(h)->ahead_cancelled == (b_first ? 2u : 0u),
                  "via %d: decided again / counted while decided (%zu, %u)", via, r.v.size(),
                  ct.order(h)->ahead_cancelled);
        }
    }

    // The cancel is acked (on_done) while the order is still in the book: the
    // venue's Delete for it comes later. The round lot ahead then cancels,
    // putting it at the front with (b) at 0% — no decision for a released
    // order. The symbol is pinned so the updates are processed.
    static void abandon_after_on_done_is_silent() {
        MiniBook bk;
        CancelParams p; p.abandon_pct = 0.0; p.queue_n = 0;
        CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
        ct.set_tracking(1, true);
        g_nbbo[1] = {100000, 100100};
        FeedTimeType t = 1'000'000;
        bk.add(1000, 1, Side::Buy, 99900, 300);
        const H h = place(ct, bk, 7, 1, Side::Buy, 99900, 100, t);
        CHECK(ct.order(h)->ahead_lot_n == 1 && ct.on_done(h), "setup");
        Rec r;
        leave(ct, bk, 1000, BookEvent::Delete, t += 1000, r.cb());
        CHECK(r.v.empty() && QBlockList::alone(bk.pool, bk.qh(7)), "decision for a released order (%zu)", r.v.size());
        CHECK(leave(ct, bk, 7, BookEvent::Delete, t += 1000, r.cb()) == (kQGone | kQTracked) && r.v.empty(),
              "its own late Delete");
    }

    // After a cancel reject, rearm: (b) fires again at the next removal at the
    // level while it still holds (an order joins behind and leaves) — not at
    // a removal at another price.
    static void abandon_rearm() {
        MiniBook bk;
        CancelParams p; p.abandon_pct = 0.0; p.queue_n = 0;                   // any cancellation
        CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        FeedTimeType t = 1'000'000;
        bk.add(1000, 1, Side::Buy, 99900, 300);
        const H h = place(ct, bk, 7, 1, Side::Buy, 99900, 100, t);
        Rec r;
        leave(ct, bk, 1000, BookEvent::Delete, t += 1000, r.cb());
        CHECK(r.v.size() == 1 && r.v[0].d == D::CancelAbandoned, "first (b)");
        CHECK(ct.rearm(h) && ct.evaluate(h) == D::CancelAbandoned, "rearm / evaluate");
        bk.add(2000, 1, Side::Buy, 99800, 300);                              // another of ours, armed, a tick lower:
        const H h2 = place(ct, bk, 9, 1, Side::Buy, 99800, 100, t);          //   a removal at ITS price
        leave(ct, bk, 2000, BookEvent::Execute, t += 1000, r.cb());          //   (0 of 1 cancelled: no (b) for it)
        CHECK(r.v.size() == 1 && ct.order(h2)->active, "fired on a removal at another price");
        bk.add(1001, 1, Side::Buy, 99900, 100);
        ct.on_book_update(1, BookEvent::Add, Side::Buy, 99900, t += 1000, 0, r.cb());
        CHECK(r.v.size() == 1, "fired on an add");
        leave(ct, bk, 1001, BookEvent::Delete, t += 1000, r.cb());
        CHECK(r.v.size() == 2 && r.v[1].d == D::CancelAbandoned, "no second (b) after rearm");
    }

    // The round lots counted for an order are those holding a round lot when
    // it is placed. R1 and R2 (300 each) are flagged for X; X leaves; R1 is
    // cut to 50. Y, placed then, counts R2 only: R1's flag is gone, its
    // delete is no cancellation, and with R2 filled Y is at the front with 0
    // of 1 — no (b) even at X = 0%. Had X still rested ahead of Y, Y would
    // have taken X's flags for the orders ahead of X: 2, R1 included.
    static void abandon_counts_round_lots_as_placed() {
        for (int shared = 0; shared < 2; ++shared) {
            MiniBook bk;
            CancelParams p; p.queue_n = 0; p.abandon_pct = 0.0;
            CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
            g_nbbo[1] = {100000, 100100};
            FeedTimeType t = 1'000'000;
            bk.add(1000, 1, Side::Buy, 99900, 300);
            bk.add(1001, 1, Side::Buy, 99900, 300);
            const H x = place(ct, bk, 7, 1, Side::Buy, 99900, 100, t);
            Rec r;
            CHECK(ct.order(x)->ahead_lot_n == 2, "X counts %u", ct.order(x)->ahead_lot_n);
            if (!shared) leave(ct, bk, 7, BookEvent::Delete, t += 1000, r.cb());
            const uint8_t f = bk.reduce(1000, 250);
            ct.on_book_update(1, BookEvent::Cancel, Side::Buy, 99900, t += 1000, f, r.cb());
            const H y = place(ct, bk, 8, 1, Side::Buy, 99900, 100, t);
            CHECK(f == kQAhead && ct.order(y)->ahead_lot_n == (shared ? 2u : 1u), "shared %d: Y counts %u", shared,
                  ct.order(y)->ahead_lot_n);
            if (shared) continue;
            CHECK(leave(ct, bk, 1000, BookEvent::Delete, t += 1000, r.cb()) == kQGone, "R1 still flagged");
            leave(ct, bk, 1001, BookEvent::Execute, t += 1000, r.cb());
            CHECK(r.v.empty() && ct.order(y)->ahead_cancelled == 0 && QBlockList::ahead_n(bk.pool, bk.qh(8)) == 0,
                  "(b) from an order that was no round lot at placement: %zu decisions", r.v.size());
        }
    }

    // (b) switched off while an order rests with its round lots counted: no
    // (b) for it, though it reaches the front (X < 0 must not read as "any
    // ratio passes"). Switched on again it stays disarmed; rearm(h) re-arms.
    static void abandon_switched_off_at_runtime() {
        MiniBook bk;
        CancelParams p; p.queue_n = 0; p.abandon_pct = 0.0;
        CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        FeedTimeType t = 1'000'000;
        bk.add(1000, 1, Side::Buy, 99900, 300);
        bk.add(1001, 1, Side::Buy, 99900, 300);
        bk.add(1002, 1, Side::Buy, 99900, 300);
        const H h = place(ct, bk, 7, 1, Side::Buy, 99900, 100, t);
        Rec r;
        p.abandon_pct = -1.0; ct.set_params(p);
        leave(ct, bk, 1000, BookEvent::Delete, t += 1000, r.cb());           // cancelled, unseen
        p.abandon_pct = 0.0; ct.set_params(p);
        leave(ct, bk, 1001, BookEvent::Delete, t += 1000, r.cb());           // still disarmed
        CHECK(r.v.empty() && ct.order(h)->ahead_cancelled == 0 && ct.evaluate(h) == D::Hold, "counted while off");
        p.abandon_pct = -1.0; ct.set_params(p);
        leave(ct, bk, 1002, BookEvent::Delete, t += 1000, r.cb());           // at the front, (b) off
        CHECK(r.v.empty() && ct.order(h)->active && ct.evaluate(h) == D::Hold, "(b) fired while off: %zu", r.v.size());
        MiniBook b2;                                                         // rearm with (b) off does not arm:
        CancelParams q; q.queue_n = 0; q.abandon_pct = 0.0;                   //   a cancellation ahead goes uncounted
        CT c2(4, 16, b2.pool, q); c2.set_symbol(1, T, T);
        b2.add(1000, 1, Side::Buy, 99900, 300);
        b2.add(1001, 1, Side::Buy, 99900, 300);
        const H g = place(c2, b2, 7, 1, Side::Buy, 99900, 100, t);
        q.abandon_pct = -1.0; c2.set_params(q);
        CHECK(c2.rearm(g), "rearm");
        leave(c2, b2, 1000, BookEvent::Delete, t += 1000, r.cb());
        CHECK(r.v.empty() && c2.order(g)->ahead_cancelled == 0, "armed by rearm with (b) off");
        q.abandon_pct = 0.0; c2.set_params(q);                               // on again, and rearm: counting resumes
        CHECK(c2.rearm(g), "rearm");
        leave(c2, b2, 1001, BookEvent::Delete, t += 1000, r.cb());
        CHECK(r.v.size() == 1 && r.v[0].h == g && r.v[0].d == D::CancelAbandoned && c2.order(g)->ahead_cancelled == 1,
              "rearm with (b) on again: %zu decisions", r.v.size());
    }

    // prefetch(sym) is a hint: for an untracked symbol, a pinned one, one
    // with an armed order, one with both sides and lanes 4-7 in use, it
    // changes no order, no tracking bit and no decision. (That the prefetches
    // are emitted at all is a property of the build: the bench's "+pf" rows
    // and the handler's machine code show it; see CancelTactic::prefetch.)
    static void prefetch_is_only_a_hint() {
        MiniBook bk;
        CancelParams p; p.queue_n = 0; p.abandon_pct = 0.0;
        CT ct(8, 32, bk.pool, p);
        for (uint16_t y = 0; y < 8; ++y) { ct.set_symbol(y, T, T); g_nbbo[y] = {100000, 100100}; }
        FeedTimeType t = 1'000'000;
        ct.set_tracking(2, true);                                            // pinned, no orders
        bk.add(1000, 1, Side::Buy, 99900, 300);
        std::vector<H> hs{place(ct, bk, 7, 1, Side::Buy, 99900, 100, t)};    // one order, (b) armed
        for (uint64_t k = 0; k < 6; ++k) {                                   // both sides, lanes 4-7 in use
            hs.push_back(place(ct, bk, 20 + k, 3, Side::Sell, 100300 + PriceType(k) * T, 100, t));
            hs.push_back(place(ct, bk, 40 + k, 3, Side::Buy,  99800 - PriceType(k) * T, 100, t));
        }
        const auto snap = [&] {
            std::vector<std::tuple<uint64_t, PriceType, uint32_t, uint32_t, uint32_t, unsigned, unsigned, bool, D>> v;
            for (H h : hs) {
                const auto o = *ct.order(h);
                v.emplace_back(o.id, o.px, o.qh, o.sz, o.queue_ahead, o.ahead_lot_n, o.ahead_cancelled, o.active,
                               ct.evaluate(h));
            }
            return v;
        };
        const auto before = snap();
        bool tr[8];
        for (uint16_t y = 0; y < 8; ++y) tr[y] = ct.tracking(y);
        for (int rep_ = 0; rep_ < 3; ++rep_)
            for (uint16_t y = 0; y < 8; ++y) ct.prefetch(y);
        bool same_tr = true;
        for (uint16_t y = 0; y < 8; ++y) same_tr = same_tr && tr[y] == ct.tracking(y);
        CHECK(snap() == before && same_tr && ct.live_orders() == hs.size(), "prefetch changed state");
        CHECK(tr[1] && tr[2] && tr[3] && !tr[0] && !tr[4], "tracking: %d %d %d %d", tr[0], tr[1], tr[2], tr[3]);
        Rec r;
        ct.prefetch(1);
        leave(ct, bk, 1000, BookEvent::Delete, t += 1000, r.cb());           // and the decision comes as before
        CHECK(r.v.size() == 1 && r.v[0].h == hs[0] && r.v[0].d == D::CancelAbandoned, "%zu decisions", r.v.size());
    }

    // rearm re-reads how many orders are ahead: departures while the order
    // was decided went unseen. R1 R2 X R3 Y, all round lots: Y's scan flags
    // R3, behind X. A bid above ours trades: (d) for X and Y (T = 1). R1
    // fills, unseen. X is rearmed: one ahead (R2). R3 cancels: behind X, not
    // X's. R2 fills: X at the front with 0 of 2 cancelled — no (b) at 40%.
    static void abandon_rearm_rereads_the_queue() {
        MiniBook bk;
        CancelParams p; p.queue_n = 0; p.abandon_pct = 40.0; p.aggr_prints_n = 1;
        CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        FeedTimeType t = 1'000'000;
        bk.add(1000, 1, Side::Buy, 99900, 200);
        bk.add(1001, 1, Side::Buy, 99900, 200);
        const H x = place(ct, bk, 7, 1, Side::Buy, 99900, 100, t);
        bk.add(1002, 1, Side::Buy, 99900, 200);
        const H y = place(ct, bk, 8, 1, Side::Buy, 99900, 100, t);
        Rec r;
        exec(ct, 1, Side::Buy, 100000, t += 1000, r.cb());
        CHECK(r.v.size() == 2 && r.v[0].d == D::CancelOutcompeted && r.v[1].d == D::CancelOutcompeted &&
              ct.order(y)->ahead_lot_n == 3, "setup: %zu decisions", r.v.size());
        leave(ct, bk, 1000, BookEvent::Execute, t += 1000, r.cb());
        CHECK(ct.rearm(x), "rearm");
        leave(ct, bk, 1002, BookEvent::Delete, t += 1000, r.cb());
        CHECK(ct.order(x)->ahead_cancelled == 0, "counted a cancellation behind X");
        leave(ct, bk, 1001, BookEvent::Execute, t += 1000, r.cb());
        CHECK(r.v.size() == 2 && ct.order(x)->active, "(b) at 0%%: %zu decisions", r.v.size());
    }

    // ---- 12. (d) from the join signal --------------------------------------------------------

    // A buy signal at 10.01; then bids above it hit three times at 10.02 and
    // an offer lifted at 10.03. With T = 3 an order placed at 10.01 or 10.00
    // starts at 3 and fires at placement; at 10.02 it starts at 0. The option
    // off, or the kill flag set (before the join tactic has read it): 0.
    // Disarmed, the signal still lives: 3. With passive_side_only off (T = 10
    // here) the offer lifted counts too: 4. A start of 3 with T = 4 waits for
    // one more execution.
    static void aggr_counts_from_the_join_signal() {
        struct V { PriceType px; bool opt, kill, disarm, passive; uint32_t want, t; };
        const V vs[] = {{100100, true, false, false, true, 3, 3}, {100000, true, false, false, true, 3, 3},
                        {100200, true, false, false, true, 0, 3}, {100100, false, false, false, true, 0, 3},
                        {100100, true, true, false, true, 0, 3},  {100100, true, false, true, true, 3, 3},
                        {100100, true, false, false, false, 4, 10}, {100100, true, false, false, true, 3, 4}};
        int i = 0;
        for (const V& v : vs) {
            g_st[1].tac = TacticShared{};
            CancelParams p; p.aggr_from_signal = v.opt; p.passive_side_only = v.passive;
            p.aggr_prints_n = v.t;
            CT ct(4, 16, g_nopool, p); ct.set_symbol(1, T, T);
            g_nbbo[1] = {100100, 100200};
            JoinTactic jt(4); jt.arm(1, true);
            int sigs = 0;
            jt.on_add(1, Side::Buy, 100100, 200, 100000, 1'000, [&](uint16_t, Side, const JoinSide&) { ++sigs; });
            for (int k = 0; k < 3; ++k) jt.on_execute(1, Side::Buy, 100200, 100);
            jt.on_execute(1, Side::Sell, 100300, 100);
            if (v.kill) JoinTactic::kill(1, Side::Buy);
            if (v.disarm) jt.arm(1, false);
            Rec r;
            const H h = ct.on_place(7, 1, Side::Buy, v.px, 2'000, NQ, r.cb());
            CHECK(sigs == 1 && ct.order(h)->n_aggr_prints == v.want, "case %d: seeded %u, want %u", i,
                  ct.order(h)->n_aggr_prints, v.want);
            CHECK((r.v.size() == 1) == (v.want >= p.aggr_prints_n) &&
                  (r.v.empty() || r.v[0].d == D::CancelOutcompeted), "case %d: %zu decisions", i, r.v.size());
            ++i;
        }
    }

    static void run_all(const char* tag) {
        run(tag, "baseline: (c) fires once the far touch settles",          baseline_depth_fires_once_far_touch_settles);
        run(tag, "baseline: (d) counts only executions against our side",   baseline_outcompeted_counts_only_executions_against_our_side);
        run(tag, "baseline: re-entry hysteresis",                           baseline_reentry_hysteresis);
        run(tag, "baseline: the execution pass on the sell side",           execution_pass_on_the_sell_side);
        run(tag, "baseline: an execution completing (d) skips (c)",         execution_that_completes_d_skips_c);
        run(tag, "baseline: hysteresis uses the threshold that fired",      hysteresis_uses_the_threshold_that_fired);
        run(tag, "baseline: lane capacity per side",                        lane_capacity_per_side);
        run(tag, "1 every book event kind triggers the check",              every_book_event_kind_triggers_the_check);
        run(tag, "1 only executions count toward (d)",                      only_executions_count_toward_b);
        run(tag, "1 unchanged far touch does not restart the gate",         unchanged_far_touch_does_not_restart_the_gate);
        run(tag, "1 degenerate NBBO from getNBBO holds (c)",                degenerate_nbbo_holds);
        run(tag, "1 sell side: far touch is the bid",                       sell_side_far_touch_is_the_bid);
        run(tag, "1 on_place baseline is a fresh getNBBO read",             on_place_baseline_is_a_fresh_nbbo);
        run(tag, "1 replace_allowed reads a fresh getNBBO",                 replace_allowed_reads_a_fresh_nbbo);
        run(tag, "2 early-stamped execution is clamped",                    early_stamped_execution_is_clamped);
        run(tag, "2 early-stamped quote keeps sigma",                       early_stamped_quote_keeps_sigma);
        run(tag, "2 stability gate holds for update stamped before move",   gate_holds_for_update_stamped_before_far_move);
        run(tag, "2 (d) ignores executions at or before placement",         outcompeted_ignores_executions_at_or_before_placement);
        run(tag, "2 hysteresis with older clock does not age out",          hysteresis_with_older_clock_does_not_age_out);
        run(tag, "3 level-clearing sells counted (NBBO already moved)",     level_clearing_nbbo_already_moved);
        run(tag, "3 level-clearing sells counted (NBBO moves after)",       level_clearing_nbbo_moves_after);
        run(tag, "3 sweep in one timestamp counts each execution",          sweep_in_one_timestamp_counts_each_execution);
        run(tag, "3 aggressor from resting side, locked/crossed NBBO",      aggressor_from_resting_side_with_locked_or_crossed_nbbo);
        run(tag, "3 passive_side_only off counts both sides",               both_sides_count_without_passive_side_only);
        run(tag, "4 callback cancel-replace does not double count",         callback_cancel_replace_does_not_double_count);
        run(tag, "4 callback freeing pending sibling drops its decision",   callback_freeing_pending_sibling_drops_its_decision);
        run(tag, "4 nested update for another symbol from callback",        nested_update_for_another_symbol_from_callback);
        run(tag, "4 nested same-symbol update completes deferred work",     nested_same_symbol_update_completes_deferred_bookkeeping);
        run(tag, "4 rearm after (d) needs one more counted execution",      rearm_after_outcompeted_needs_one_more_execution);
        run(tag, "5 too-late-to-cancel race keeps new order",               too_late_to_cancel_race_keeps_new_order);
        run(tag, "5 pool exhaustion and reuse",                             pool_exhaustion_and_reuse);
        run(tag, "6 unchanged NBBO does no estimator work",                 unchanged_nbbo_does_no_estimator_work);
        run(tag, "6 trade gap seeded with first gap",                       trade_gap_seeded_with_first_gap);
        run(tag, "6 same-timestamp executions are one trade event",         same_timestamp_executions_are_one_event);
        run(tag, "6 same-timestamp mid burst records net move",             same_timestamp_mid_burst_records_net_move);
        run(tag, "7 hist spread floor is fixed per symbol",                 hist_spread_floor_is_fixed_per_symbol);
        run(tag, "8 untracked symbol costs one bit test",                   untracked_symbol_costs_one_bit_test);
        run(tag, "8 tracking follows active orders",                        tracking_follows_active_orders);
        run(tag, "8 pin keeps estimators warm",                             pin_keeps_estimators_warm);
        run(tag, "8 resuming restarts the trade gap",                       resume_restarts_the_trade_gap);
        run(tag, "9 on_place flags the queue slot; size/position from book", on_place_flags_the_queue_slot);
        run(tag, "9 book remove releases the lane",                         book_remove_releases_the_lane);
        run(tag, "9 flag after on_done is harmless",                        flag_after_on_done_is_harmless);
        run(tag, "9 missed remove never reads another order's slot",       missed_remove_never_reads_another_order);
        run(tag, "9 (d) queue gate reads position from the book",           queue_gate_reads_position_from_book);
        run(tag, "9 spilled level: position across blocks",                 spilled_level_queue_position);
        run(tag, "10 (a) fires at placement, from on_place",                queue_fires_at_placement);
        run(tag, "10 (a) needs N orders and K x our size ahead",            queue_needs_both_limits);
        run(tag, "10 (a) defaults: N = 5, K = 3",                           queue_defaults_five_and_three);
        run(tag, "10 (a) re-checked when our order is reduced",             queue_rechecked_when_ours_is_reduced);
        run(tag, "10 (a) decision dropped when freed before delivery",      queue_decision_dropped_when_freed_first);
        run(tag, "11 (b) fires when ours reaches the front",                abandon_fires_at_the_front);
        run(tag, "11 (b) needs more than X% cancelled",                     abandon_needs_more_than_x);
        run(tag, "11 (b) ignores what is behind; partial cancels do not count", abandon_ignores_what_is_behind);
        run(tag, "11 (b) each order of ours, in queue order",               abandon_each_order_in_queue_order);
        run(tag, "11 (b) re-fires after rearm while it holds",              abandon_rearm);
        run(tag, "11 (b) rearm re-reads the orders ahead",                  abandon_rearm_rereads_the_queue);
        run(tag, "11 (b) switched off at runtime: disarmed, never fires",    abandon_switched_off_at_runtime);
        run(tag, "11 (b) counts the round lots as they are at placement",    abandon_counts_round_lots_as_placed);
        run(tag, "11 (b) counts at its own level only",                     abandon_counts_its_own_level_only);
        run(tag, "11 a decided order is decided once",                      decided_order_is_decided_once);
        run(tag, "11 (b) silent after on_done, order still in the book",     abandon_after_on_done_is_silent);
        run(tag, "12 (d) counts from the join signal, at any price",        aggr_counts_from_the_join_signal);
        run(tag, "13 prefetch(sym) is only a hint",                         prefetch_is_only_a_hint);
    }
};

// ---- volatility from the venue's round-lot BBO (VolSource::Venue) ---------------------------------

template <bool kSimd>
struct VenueSuite {
    using V = CancelTacticT<kSimd, VolSource::Venue>;
    using N = CancelTacticT<kSimd, VolSource::Nbbo>;
    using H = typename V::Handle;
    static constexpr auto nop = [](auto, uint64_t, auto) {};
    template <class CT>
    static void upd(CT& ct, uint16_t s, FeedTimeType ts) {
        ct.on_book_update(s, BookEvent::Modify, Side::Buy, 0, ts, 0, nop);
    }

    static void ignores_the_nbbo_mid() {
        V ct(4, 16, g_nopool); ct.set_symbol(1, T, T); ct.set_tracking(1, true);
        g_venue[1] = {100000, 100100};
        FeedTimeType t = 1'000'000'000;
        for (int k = 0; k < 200; ++k) {
            t += 10'000'000;
            g_nbbo[1] = {100000 + (k & 1) * T, 100100 + (k & 1) * T};
            upd(ct, 1, t);
        }
        CHECK(ct.sigma_rate(1) == 0.0, "the NBBO mid moved sigma: %.4g", ct.sigma_rate(1));
        for (int k = 0; k < 200; ++k) {
            t += 10'000'000;
            g_venue[1] = {100000 + (k & 1) * T, 100100 + (k & 1) * T};
            upd(ct, 1, t);
        }
        CHECK(ct.sigma_rate(1) > 0.0, "the venue mid did not move sigma");
    }

    // The same mid path through the venue hook (NBBO frozen) and through
    // the NBBO (default build) gives the same sigma, bit for bit, including
    // same-timestamp bursts, early stamps and one-sided stretches.
    static void same_estimator_as_the_nbbo_build() {
        V v(4, 16, g_nopool); N n(4, 16, g_nopool);
        v.set_symbol(1, T, T); n.set_symbol(1, T, T);
        v.set_tracking(1, true); n.set_tracking(1, true);
        std::mt19937_64 rng(5);
        FeedTimeType t = 1'000'000'000;
        PriceType b = 100000, a = 100100;
        int mismatch = 0;
        for (int k = 0; k < 20'000; ++k) {
            if (rng() % 100 < 80) t += rng() % 50'000'000;            // else the same timestamp
            FeedTimeType ts = t;
            if (rng() % 100 < 3) ts = t - rng() % 1'000'000;          // early stamp
            const unsigned m = static_cast<unsigned>(rng() % 100);
            if (m < 35)      { const PriceType st = (rng() & 1) ? T : -T; b += st; }
            else if (m < 70) { const PriceType st = (rng() & 1) ? T : -T; a += st; }
            if (a <= b) a = b + T;
            Nbbo q{b, a};
            const unsigned deg = static_cast<unsigned>(rng() % 100);
            if (deg < 3) q.bid = 0; else if (deg < 6) q.ask = 0;     // no round lot on one side
            g_venue[1] = q; g_nbbo[1] = {100000, 100100};             // venue build: the path is the venue's
            upd(v, 1, ts);
            g_nbbo[1] = q;                                            // NBBO build: the same path
            upd(n, 1, ts);
            mismatch += v.sigma_rate(1) != n.sigma_rate(1);
        }
        CHECK(mismatch == 0 && v.sigma_rate(1) > 0.0, "%d of 20000 updates differ", mismatch);
    }

    // Round lots, from a book: an odd lot improving the raw bid leaves the
    // round-lot bid, and sigma, alone; the round lot completing at that price
    // moves both.
    static void odd_lots_do_not_move_it() {
        MiniBook bk;
        g_vbook = &bk;
        V ct(4, 16, bk.pool); ct.set_symbol(1, T, T); ct.set_tracking(1, true);
        g_nbbo[1] = {100000, 100300};
        FeedTimeType t = 1'000'000'000;
        bk.add(1, 1, Side::Buy, 100000, 200);
        bk.add(2, 1, Side::Sell, 100300, 300);
        upd(ct, 1, t);
        t += 1'000'000'000; bk.add(3, 1, Side::Buy, 100100, 50);       // odd lot at 10.01
        upd(ct, 1, t);
        CHECK(bk.rl_bbo(1).bid == 100000 && ct.sigma_rate(1) == 0.0,
              "an odd lot moved the round-lot bid (%lld) or sigma (%.4g)", (long long)bk.rl_bbo(1).bid,
              ct.sigma_rate(1));
        t += 1'000'000'000; bk.add(4, 1, Side::Buy, 100100, 60);       // 110 shares at 10.01
        upd(ct, 1, t);
        CHECK(bk.rl_bbo(1).bid == 100100 && ct.sigma_rate(1) > 0.0, "a round lot at 10.01 did not move sigma");
        t += 1'000'000'000; bk.reduce(4, 59);                           // back to 51 shares: 10.00 again
        upd(ct, 1, t);
        CHECK(bk.rl_bbo(1).bid == 100000, "round-lot bid after the reduce: %lld", (long long)bk.rl_bbo(1).bid);
        g_vbook = nullptr;
    }

    // Depth, the stability gate and the floors stay on the NBBO: the venue's
    // own book sits a dollar away and never moves, and (c) fires exactly as in
    // the default build, on a threshold with no vol term.
    static void decisions_stay_on_the_nbbo() {
        V ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        g_venue[1] = {99000, 99100};
        FeedTimeType u = 2'000'000'000;
        ct.set_tracking(1, true);
        g_nbbo[1] = {100000, 100100}; upd(ct, 1, u);
        for (int k = 0; k < 30; ++k) { u += 1'000'000; ct.on_book_update(1, BookEvent::Execute, Side::Buy, 100000, u, 0, nop); }
        int fired = 0; H fh = 0;
        auto cb = [&](H h, uint64_t, auto) { ++fired; fh = h; };
        const H h = ct.on_place(2, 1, Side::Buy, 99800, u, V::kNoQueue, nop);
        ct.set_tracking(1, false);
        u += 1'000'000; g_nbbo[1] = {100000, 101000};
        ct.on_book_update(1, BookEvent::Add, Side::Sell, 101000, u, 0, cb);
        ct.on_book_update(1, BookEvent::Modify, Side::Buy, 0, u + 100'000, 0, cb);
        CHECK(fired == 0, "fired inside the NBBO far-touch window");
        ct.on_book_update(1, BookEvent::Modify, Side::Buy, 0, u + 300'000, 0, cb);
        CHECK(fired == 1 && fh == h, "depth against the NBBO did not fire (n=%d)", fired);
        CHECK(ct.sigma_rate(1) == 0.0 && ct.threshold(1) == 1000, "sigma %.4g, threshold %lld",
              ct.sigma_rate(1), (long long)ct.threshold(1));
    }

    // The round-lot quote is read once per tracked update, after the decision
    // pass and its callbacks: never on the path to a cancel, never for an
    // untracked symbol.
    static void hook_is_off_the_decision_path() {
        V ct(4, 16, g_nopool); ct.set_symbol(1, T, T);
        g_venue[1] = {100000, 100100}; g_nbbo[1] = {100000, 100100};
        FeedTimeType u = 2'000'000'000;
        const uint64_t r0 = g_venue_reads;
        upd(ct, 1, u);
        CHECK(g_venue_reads == r0, "untracked update read the venue quote");
        ct.set_tracking(1, true);
        for (int k = 0; k < 30; ++k) { u += 1'000'000; ct.on_book_update(1, BookEvent::Execute, Side::Buy, 100000, u, 0, nop); }
        CHECK(g_venue_reads == r0 + 30, "30 tracked updates, %llu reads", (unsigned long long)(g_venue_reads - r0));
        ct.on_place(2, 1, Side::Buy, 99800, u, V::kNoQueue, nop);
        ct.set_tracking(1, false);                                      // tracked by the order now
        u += 1'000'000; g_nbbo[1] = {100000, 101000};
        upd(ct, 1, u);                                                  // offer gaps: (c) due at +200 us
        int fired = 0; uint64_t at_cb = 0;
        auto cb = [&](H, uint64_t, auto) { ++fired; at_cb = g_venue_reads; };
        const uint64_t r1 = g_venue_reads;
        ct.on_book_update(1, BookEvent::Modify, Side::Buy, 0, u + 300'000, 0, cb);
        CHECK(fired == 1, "(c) did not fire");
        CHECK(at_cb == r1 && g_venue_reads == r1 + 1, "%llu reads before the callback, %llu in all",
              (unsigned long long)(at_cb - r1), (unsigned long long)(g_venue_reads - r1));
    }

    // A callback that feeds an update for the same symbol: that update first
    // completes the deferred one's sample, from the book as it then stands.
    // Later timestamp: the same as sequential processing in which the first
    // update already sees the second's round-lot quote, bit for bit. Same
    // timestamp: the same as plain sequential processing — the same net move,
    // which the sequential path forms as a same-instant revision (so equal to
    // rounding).
    static void nested_update_completes_deferred_work() {
        auto setup = [](V& ct, FeedTimeType& u) {
            ct.set_symbol(1, T, T);
            u = 2'000'000'000;
            ct.set_tracking(1, true);                                   // stays pinned: nested update runs
            g_venue[1] = {100000, 100100}; g_nbbo[1] = {100000, 100100};
            upd(ct, 1, u);
            for (int k = 0; k < 30; ++k) {
                u += 1'000'000;
                g_venue[1] = {100000 + (k & 1) * T, 100100 + (k & 1) * T};
                ct.on_book_update(1, BookEvent::Execute, Side::Buy, 100000, u, 0, nop);
            }
            ct.on_place(2, 1, Side::Buy, 99800, u, V::kNoQueue, nop);
            u += 1'000'000; g_nbbo[1] = {100000, 101000};
            upd(ct, 1, u);                                              // offer gaps: (c) due at +200 us
        };
        struct Cb {
            V& ct; FeedTimeType t; int n = 0;
            void operator()(H, uint64_t, typename V::Decision) {
                ++n;
                g_venue[1] = {100300, 100400};                          // the host applies the next update
                ct.on_book_update(1, BookEvent::Execute, Side::Sell, 100400, t, 0, nop);
            }
        };
        for (const FeedTimeType dt : {FeedTimeType{1'000}, FeedTimeType{0}}) {
            V a(4, 16, g_nopool), c(4, 16, g_nopool);
            FeedTimeType ua, uc;
            setup(a, ua); setup(c, uc);
            Cb cb{a, ua + 300'000 + dt};
            g_venue[1] = {100200, 100300};
            a.on_book_update(1, BookEvent::Execute, Side::Sell, 101000, ua + 300'000, 0, cb);
            g_venue[1] = dt ? Nbbo{100300, 100400} : Nbbo{100200, 100300};
            c.on_book_update(1, BookEvent::Execute, Side::Sell, 101000, uc + 300'000, 0, nop);
            g_venue[1] = {100300, 100400};
            c.on_book_update(1, BookEvent::Execute, Side::Sell, 100400, uc + 300'000 + dt, 0, nop);
            CHECK(cb.n == 1, "callbacks=%d", cb.n);
            const double sa = a.sigma_rate(1), sc = c.sigma_rate(1);
            CHECK((dt ? sa == sc : std::fabs(sa - sc) <= 1e-12 * sc) && sc > 0.0 &&
                  a.trade_gap_ns(1) == c.trade_gap_ns(1) && a.threshold(1) == c.threshold(1),
                  "dt=%llu: nested sigma %.17g vs sequential %.17g", (unsigned long long)dt, sa, sc);
        }
    }

    static void run_all(const char* tag) {
        run(tag, "venue vol: ignores the NBBO mid",                          ignores_the_nbbo_mid);
        run(tag, "venue vol: same estimator as the NBBO build, bit for bit",  same_estimator_as_the_nbbo_build);
        run(tag, "venue vol: odd lots do not move it (round-lot BBO)",        odd_lots_do_not_move_it);
        run(tag, "venue vol: depth, gate and floors stay on the NBBO",        decisions_stay_on_the_nbbo);
        run(tag, "venue vol: round-lot quote read after the callbacks",       hook_is_off_the_decision_path);
        run(tag, "venue vol: nested update completes deferred work",          nested_update_completes_deferred_work);
    }
};

// ---- exp2 kernel ---------------------------------------------------------------------------

static void exp2_accuracy() {
    double worst = 0.0, at = 0.0;
    auto probe = [&](double x) {
        const double a = ct_detail::exp2_neg(x), e = std::exp2(x);
        const double rel = std::fabs(a - e) / e;
        if (rel > worst) { worst = rel; at = x; }
    };
    for (int i = 0; i <= 2'000'000; ++i) probe(-256.0 * i / 2'000'000.0);       // full range
    for (int i = 1; i <= 1'000'000; ++i) probe(-std::ldexp(1.0, -40) * i);      // dt << half-life
    CHECK(worst < 1e-10, "max relative error %.3g at x=%.6g", worst, at);
    CHECK(ct_detail::exp2_neg(0.0) == 1.0, "exp2(0) = %.17g", ct_detail::exp2_neg(0.0));
    CHECK(ct_detail::exp2_neg(-1e12) == std::ldexp(1.0, -256), "clamp below -256");
    double dw = 0.0, dat = 0.0;                                  // decay(): both sides of the switch
    const double inv = 1.0 / 60e9;                               // 1-minute half-life
    for (int i = 0; i <= 400'000; ++i) {
        const uint64_t dt = static_cast<uint64_t>(std::pow(10.0, 13.0 * i / 400'000.0));   // 1 ns .. 3 h
        const double a = ct_detail::decay(dt, inv), e = std::exp2(-static_cast<double>(dt) * inv);
        const double rel = e > 0 ? std::fabs(a - e) / e : 0.0;
        if (static_cast<double>(dt) * inv < 250.0 && rel > dw) { dw = rel; dat = static_cast<double>(dt); }
    }
    CHECK(dw < 1e-10, "decay max relative error %.3g at dt=%.6g ns", dw, dat);
    CHECK(ct_detail::decay(0, inv) == 1.0, "decay(0) = %.17g", ct_detail::decay(0, inv));
    std::printf("      exp2_neg max relative error %.2e, decay %.2e\n", worst, dw);
}

// ---- QBlockList against a reference model ---------------------------------------------------

static void qblocklist_layout() {
    CHECK(sizeof(QBlockPool::Block) == 224 && alignof(QBlockPool::Block) == 32 && sizeof(QBlockList) == 8,
          "layout");
    QBlockPool p(64);
    int straddle = 0, misaligned = 0;
    for (uint32_t i = 1; i <= 64; ++i) {
        const auto a = reinterpret_cast<uintptr_t>(&p.at(i));
        misaligned += (a % 32) != 0;
        straddle   += (a % 64) + 32 > 64;
    }
    CHECK(misaligned == 0 && straddle == 0, "%d misaligned blocks, %d headers straddling a line",
          misaligned, straddle);
}

// Random push / remove / reduce / track on several lists sharing one pool,
// against a vector-per-list model in priority order: ahead, ahead_of,
// position, size_of, tracked, remove's report, for_each order, and the
// invariants trk subset of occ, live == popcount(occ), bsz == sum.
static void qblocklist_model() {
    struct M { uint64_t id; uint32_t sz; bool trk; uint32_t h; bool ahd = false; };
    constexpr unsigned NL = 6;
    QBlockPool p(512);
    QBlockList q[NL];
    std::vector<M> m[NL];
    std::mt19937_64 rng(99);
    auto U = [&](uint64_t n) { return rng() % n; };
    uint64_t next_id = 1;
    long ops = 0, bad = 0, spills = 0, scans = 0;
    auto check_list = [&](unsigned li) {
        std::size_t k = 0;
        bool order_ok = true;
        q[li].for_each(p, [&](uint32_t h, uint64_t id, uint32_t sz) {
            if (k >= m[li].size() || m[li][k].id != id || m[li][k].sz != sz || m[li][k].h != h) order_ok = false;
            ++k; return true;
        });
        if (!order_ok || k != m[li].size()) ++bad;
        uint32_t n = 0; uint64_t s = 0, tot = 0;
        for (const M& x : m[li]) tot += x.sz;
        for (const M& x : m[li]) {
            const QAhead a = q[li].ahead(p, x.h), b = QBlockList::ahead_of(p, x.h);
            const QPos2 pos = q[li].position(p, x.h, uint32_t(m[li].size()), tot);
            if (a.n != n || a.sz != s || b.n != n || b.sz != s) ++bad;
            if (pos.behind_n != m[li].size() - n - 1 || pos.behind_sz != tot - s - x.sz) ++bad;
            if (QBlockList::size_of(p, x.h) != x.sz || QBlockList::tracked(p, x.h) != x.trk) ++bad;
            if (QBlockList::alone(p, x.h) != (m[li].size() == 1)) ++bad;
            if (QBlockList::ahead_n(p, x.h) != n) ++bad;
            ++n; s += x.sz;
        }
    };
    long exhausted = 0;
    for (int it = 0; it < 200'000; ++it) {
        const unsigned li = static_cast<unsigned>(U(NL));
        auto& v = m[li];
        const unsigned op = static_cast<unsigned>(U(100));
        const unsigned grow = v.size() < 40 ? 55 : 25;                  // sizes hover around 40:
        if (op < grow || v.empty()) {                                   //   chains form and drain
            const uint32_t sz = static_cast<uint32_t>(1 + U(1000));
            const uint32_t h = q[li].push(p, next_id, sz);
            if (h == QBlockList::kBadHandle) { ++exhausted; continue; }
            v.push_back({next_id++, sz, false, h});
            spills += v.size() > 16;                                    // push into a chained queue
        } else {
            const std::size_t i = U(v.size());
            if (op < grow + (100 - grow) * 5 / 10) {
                const uint8_t f = q[li].remove(p, v[i].h);
                if (f != (kQGone | (v[i].trk ? kQTracked : 0) | (v[i].ahd ? kQAhead : 0))) ++bad;
                v.erase(v.begin() + std::ptrdiff_t(i));
            } else if (op < grow + (100 - grow) * 6 / 10) {                // scan ahead of a random order
                const uint32_t lot = static_cast<uint32_t>(50 + U(900));
                const bool mark = U(2) != 0;
                const QScan sc = QBlockList::scan_ahead(p, v[i].h, lot, mark);
                uint32_t n_lot = 0; uint64_t sz = 0;
                bool exact = i + 1 == v.size();                  // from the tail: flags rewritten up to
                for (std::size_t k = i; k-- > 0;) {              //   the nearest tracked order ahead,
                    sz += v[k].sz;                               //   only added from there on; never
                    if (v[k].trk) { exact = false; continue; }   //   touched on a tracked order
                    const bool f = v[k].sz >= lot || (!exact && v[k].ahd);
                    n_lot += f;
                    if (mark) v[k].ahd = f;
                }
                if (sc.n != i || sc.sz != sz || sc.n_lot != n_lot) ++bad;
                ++scans;
            } else if (op < grow + (100 - grow) * 7 / 10) {
                if (v[i].sz > 1) {
                    const uint32_t d = static_cast<uint32_t>(1 + U(v[i].sz - 1));
                    if (q[li].reduce(p, v[i].h, d) != ((v[i].trk ? kQTracked : 0) | (v[i].ahd ? kQAhead : 0))) ++bad;
                    v[i].sz -= d;
                }
            } else if (op < grow + (100 - grow) * 8 / 10) {                // grows in place
                const uint32_t d = static_cast<uint32_t>(1 + U(500));
                if (q[li].grow(p, v[i].h, d) != (kQGrew | (v[i].trk ? kQTracked : 0) | (v[i].ahd ? kQAhead : 0)))
                    ++bad;
                v[i].sz += d;
            } else {
                const bool on = U(2) != 0;
                QBlockList::track(p, v[i].h, on);
                v[i].trk = on;
            }
        }
        ++ops;
        if ((it & 63) == 0) check_list(li);
    }
    for (unsigned li = 0; li < NL; ++li) check_list(li);
    long inv = 0;
    for (uint32_t i = 1; i <= p.capacity(); ++i) {
        const auto& b = p.at(i);
        if ((b.trk | b.ahd) & ~b.occ) ++inv;
    }
    CHECK(bad == 0 && inv == 0, "%ld mismatches, %ld blocks with trk/ahd outside occ", bad, inv);
    CHECK(scans > 10'000, "%ld scans", scans);
    CHECK(spills > 10'000 && exhausted == 0, "%ld pushes into chains, %ld pool exhaustions", spills, exhausted);
    std::printf("      %ld ops on %u lists, %ld pushes into chained queues, %zu pool blocks in use\n",
                ops, NL, spills, p.in_use());
}

// ---- join tactic -------------------------------------------------------------------------------

struct JoinSuite {
    using Sig = std::vector<std::pair<Side, JoinSide>>;
    static auto rec(Sig& v) { return [&v](uint16_t, Side sd, const JoinSide& js) { v.push_back({sd, js}); }; }

    // One buy order of a full round lot above the best bid: a signal, stored
    // in SymbolState with this venue's round-lot far touch.
    static void signal_on_round_lot_improvement() {
        JoinTactic jt(4); jt.arm(1, true);
        g_rl[1][1] = RlQuote{100300, 250};                               // offer side's round-lot best
        Sig got;
        jt.on_add(1, Side::Buy, 100100, 100, 100000, 5'000, rec(got));
        const JoinSide& js = jt.signal(1, Side::Buy);
        CHECK(got.size() == 1 && got[0].first == Side::Buy, "signals %zu", got.size());
        CHECK(js.px == 100100 && js.sz == 100 && js.t == 5'000 && js.far_px == 100300 && js.far_sz == 250 &&
              js.seq == 1 && js.lot == 100, "stored px %lld sz %u far %lld/%u seq %u",
              (long long)js.px, js.sz, (long long)js.far_px, js.far_sz, js.seq);
        CHECK(&js == &g_st[1].tac.side[0] && g_st[1].tac.side[1].px == 0, "not in SymbolState::tac");
    }

    // No signal: an odd lot, an order at or behind the best, an empty side,
    // an unarmed symbol, a sell above the best offer. The far touch is not
    // read for any of them.
    static void no_signal_cases() {
        JoinTactic jt(4); jt.arm(1, true);
        Sig got;
        const uint64_t r0 = g_rl_reads;
        jt.on_add(1, Side::Buy, 100100,  99, 100000, 1, rec(got));
        jt.on_add(1, Side::Buy, 100000, 500, 100000, 1, rec(got));
        jt.on_add(1, Side::Buy,  99900, 500, 100000, 1, rec(got));
        jt.on_add(1, Side::Buy, 100100, 500,      0, 1, rec(got));
        jt.on_add(2, Side::Buy, 100100, 500, 100000, 1, rec(got));
        jt.on_add(1, Side::Sell, 100100, 500, 100000, 1, rec(got));
        CHECK(got.empty() && g_rl_reads == r0 && jt.signal(1, Side::Buy).px == 0, "%zu signals", got.size());
    }

    // Sell side: a lower offer. arm() copies the symbol's round lot.
    static void sell_side_signal_uses_the_round_lot() {
        g_st[1].round_lot = 200;
        JoinTactic jt(4); jt.arm(1, true);
        g_rl[1][0] = RlQuote{99900, 400};
        Sig got;
        jt.on_add(1, Side::Sell, 100050, 150, 100100, 1, rec(got));        // under the 200 lot
        jt.on_add(1, Side::Sell, 100050, 200, 100100, 2, rec(got));
        const JoinSide& js = jt.signal(1, Side::Sell);
        CHECK(got.size() == 1 && got[0].first == Side::Sell && js.px == 100050 && js.far_px == 99900 &&
              js.far_sz == 400 && js.lot == 200 && js.t == 2, "signals %zu px %lld", got.size(), (long long)js.px);
    }

    // After a buy signal at 10.01: executions at 10.01 (either side), beyond it
    // on the bid side, beyond it on the offer side; below it nothing. A sell
    // signal counts the mirror image.
    static void counts_executions_at_and_through() {
        JoinTactic jt(4); jt.arm(1, true);
        Sig got;
        jt.on_add(1, Side::Buy, 100100, 100, 100000, 1, rec(got));
        jt.on_execute(1, Side::Buy, 100100, 30);
        jt.on_execute(1, Side::Buy, 100200, 40);
        jt.on_execute(1, Side::Sell, 100300, 50);
        jt.on_execute(1, Side::Buy, 100000, 60);
        jt.on_execute(1, Side::Sell, 100100, 10);
        const JoinSide& b = jt.signal(1, Side::Buy);
        CHECK(b.n_at == 2 && b.sh_at == 40 && b.n_thru_same == 1 && b.sh_thru_same == 40 &&
              b.n_thru_opp == 1 && b.sh_thru_opp == 50, "buy: at %u/%u same %u/%u opp %u/%u",
              b.n_at, b.sh_at, b.n_thru_same, b.sh_thru_same, b.n_thru_opp, b.sh_thru_opp);
        jt.on_add(1, Side::Sell, 100500, 300, 100600, 2, rec(got));
        jt.on_execute(1, Side::Sell, 100400, 10);                           // through the sell signal, offer side
        jt.on_execute(1, Side::Buy, 100450, 20);                            // through it, bid side
        jt.on_execute(1, Side::Sell, 100500, 5);                            // at it
        jt.on_execute(1, Side::Sell, 100700, 7);                            // short of it
        const JoinSide& s = jt.signal(1, Side::Sell);
        CHECK(s.n_at == 1 && s.sh_at == 5 && s.n_thru_same == 1 && s.sh_thru_same == 10 &&
              s.n_thru_opp == 1 && s.sh_thru_opp == 20, "sell: at %u same %u opp %u", s.n_at, s.n_thru_same,
              s.n_thru_opp);
        CHECK(b.n_thru_same == 2 && b.n_thru_opp == 4, "buy side kept counting: same %u opp %u",
              b.n_thru_same, b.n_thru_opp);
    }

    // While a signal lives, a new improved level is not a new signal: no
    // callback, the price, sequence and counts stay.
    static void new_levels_are_not_new_signals() {
        JoinTactic jt(4); jt.arm(1, true);
        Sig got;
        jt.on_add(1, Side::Buy, 100100, 100, 100000, 1, rec(got));
        jt.on_execute(1, Side::Buy, 100100, 30);
        jt.on_add(1, Side::Buy, 100200, 500, 100100, 2, rec(got));          // a better, bigger bid
        const JoinSide& b = jt.signal(1, Side::Buy);
        CHECK(got.size() == 1 && b.px == 100100 && b.seq == 1 && b.n_at == 1 && b.t == 1 && jt.live(1, Side::Buy),
              "re-signalled: %zu signals, px %lld seq %u", got.size(), (long long)b.px, unsigned(b.seq));
    }

    // The kill flag ends the signal when the join tactic reads it: on the
    // next execution (which then counts nothing), add or poll(). The cancel
    // tactic's count stops at once. An ended side counts nothing more while
    // the other side's signal lives. The next improvement signals again,
    // from fresh counts; a kill flag left over from before is cleared then.
    static void kill_ends_the_signal_when_read() {
        JoinTactic jt(4); jt.arm(1, true); jt.arm(2, true);
        Sig got;
        jt.on_add(1, Side::Buy, 100100, 100, 100000, 1, rec(got));
        jt.on_execute(1, Side::Buy, 100200, 30);
        const JoinSide& b = jt.signal(1, Side::Buy);
        CHECK(jt.aggr_since(1, Side::Buy, 100100, true) == 1, "setup");
        JoinTactic::kill(1, Side::Buy);
        CHECK(b.px == 100100 && jt.live(1, Side::Buy) && jt.aggr_since(1, Side::Buy, 100100, true) == 0,
              "before the read: px %lld, aggr %u", (long long)b.px, jt.aggr_since(1, Side::Buy, 100100, true));
        jt.on_execute(1, Side::Buy, 100300, 30);                            // reads the flag: ends
        CHECK(b.px == 0 && !jt.live(1, Side::Buy) && b.kill == 0 && b.n_thru_same == 1, "after the read: px %lld",
              (long long)b.px);
        jt.on_add(1, Side::Sell, 100800, 100, 100900, 2, rec(got));        // the offer side signals
        jt.on_execute(1, Side::Buy, 100300, 30);
        CHECK(got.size() == 2 && b.n_thru_same == 1 && b.n_at == 0 && jt.live(1, Side::Sell),
              "counting after the kill: %u through", b.n_thru_same);
        jt.on_add(1, Side::Buy, 100300, 200, 100200, 9, rec(got));         // signals again
        CHECK(got.size() == 3 && b.px == 100300 && b.seq == 2 && b.n_thru_same == 0 && b.t == 9 &&
              jt.aggr_since(1, Side::Buy, 100000, false) == 0, "second signal: px %lld seq %u",
              (long long)b.px, unsigned(b.seq));
        JoinTactic::kill(1, Side::Buy);                                     // read by an add that improves:
        jt.on_add(1, Side::Buy, 100400, 300, 100300, 10, rec(got));        //   ends, and that add signals
        CHECK(got.size() == 4 && b.px == 100400 && b.seq == 3 && b.kill == 0, "kill read by an add");
        JoinTactic::kill(1, Side::Buy);
        jt.poll(1);
        CHECK(b.px == 0 && !jt.live(1, Side::Buy) && jt.live(1, Side::Sell), "poll()");
        JoinTactic::kill(2, Side::Sell);                                    // no live signal there
        jt.on_add(2, Side::Sell, 100800, 100, 100900, 11, rec(got));
        jt.on_execute(2, Side::Sell, 100800, 10);
        CHECK(got.size() == 5 && jt.live(2, Side::Sell) && jt.signal(2, Side::Sell).n_at == 1,
              "a stale kill flag ended the next signal");
    }

    // Disarming stops new signals only: the live one keeps counting until
    // killed. A symbol never armed counts nothing.
    static void disarm_stops_new_signals_only() {
        JoinTactic jt(4); jt.arm(1, true);
        Sig got;
        jt.on_add(1, Side::Buy, 100100, 100, 100000, 1, rec(got));
        jt.arm(1, false);
        jt.on_execute(1, Side::Buy, 100200, 10);
        jt.on_add(1, Side::Sell, 100300, 500, 100400, 2, rec(got));        // would signal if armed
        const JoinSide& b = jt.signal(1, Side::Buy);
        CHECK(got.size() == 1 && b.px == 100100 && b.n_thru_same == 1 && jt.live(1, Side::Buy) &&
              !jt.live(1, Side::Sell), "after disarm: %zu signals, px %lld, %u through", got.size(),
              (long long)b.px, b.n_thru_same);
        JoinTactic::kill(1, Side::Buy);
        jt.on_execute(1, Side::Buy, 100200, 10);
        CHECK(b.px == 0 && b.n_thru_same == 1, "kill on a disarmed symbol");
        jt.on_execute(2, Side::Buy, 100200, 10);
        CHECK(g_st[2].tac.side[0].n_thru_same == 0 && g_st[2].tac.side[0].n_at == 0 &&
              g_st[2].tac.tape[0].same[0] == kTapeEmpty, "never-armed symbol counted");
    }

    // The tape answers "executions more aggressive than px since the signal"
    // for any px: a buy signal at 10.01, then executions against bids at
    // 10.03 10.02 10.01 10.00 9.99 and against offers at 10.04 10.02.
    static void tape_counts_beyond_any_price() {
        JoinTactic jt(4); jt.arm(1, true);
        Sig got;
        jt.on_add(1, Side::Buy, 100100, 100, 100000, 1, rec(got));
        for (PriceType x : {100300, 100200, 100100, 100000, 99900}) jt.on_execute(1, Side::Buy, x, 10);
        for (PriceType x : {100400, 100200}) jt.on_execute(1, Side::Sell, x, 10);
        struct Q { PriceType px; uint32_t same, both; };
        const Q qs[] = {{99800, 5, 7}, {100000, 3, 5}, {100100, 2, 4}, {100200, 1, 2}, {100300, 0, 1}, {100400, 0, 0}};
        int bad = 0;
        for (const Q& q : qs)
            bad += jt.aggr_since(1, Side::Buy, q.px, true) != q.same || jt.aggr_since(1, Side::Buy, q.px, false) != q.both;
        CHECK(bad == 0, "%d of 6 prices wrong (e.g. 10.00: %u / %u)", bad, jt.aggr_since(1, Side::Buy, 100000, true),
              jt.aggr_since(1, Side::Buy, 100000, false));
        // the sell side mirrors it: a sell signal at 10.10, offers lifted at 10.08, 10.10, 10.12
        jt.on_add(1, Side::Sell, 101000, 100, 101100, 2, rec(got));
        for (PriceType x : {100800, 101000, 101200}) jt.on_execute(1, Side::Sell, x, 10);
        CHECK(jt.aggr_since(1, Side::Sell, 101200, true) == 2 && jt.aggr_since(1, Side::Sell, 101000, true) == 1 &&
              jt.aggr_since(1, Side::Sell, 100800, true) == 0, "sell side: %u %u %u",
              jt.aggr_since(1, Side::Sell, 101200, true), jt.aggr_since(1, Side::Sell, 101000, true),
              jt.aggr_since(1, Side::Sell, 100800, true));
        CHECK(jt.aggr_since(1, Side::Buy, 99800, true) == 5, "the sell signal's executions reached the buy tape");
    }

    // More than 16 executions: the tape keeps exactly the 16 most aggressive,
    // sorted, and every count is exact up to 16, at every price.
    static void tape_keeps_the_most_aggressive() {
        JoinTactic jt(4); jt.arm(1, true);
        Sig got;
        jt.on_add(1, Side::Buy, 100000, 100, 99900, 1, rec(got));
        std::mt19937_64 rng(5);
        std::vector<PriceType> same, opp;
        for (int i = 0; i < 300; ++i) {
            const PriceType x = 100000 + (PriceType(rng() % 41) - 20) * T;
            const bool bid = rng() % 3 != 0;
            jt.on_execute(1, bid ? Side::Buy : Side::Sell, x, 1);
            (bid ? same : opp).push_back(x);
        }
        int bad = 0;
        for (PriceType px = 100000 - 22 * T; px <= 100000 + 22 * T; px += T) {
            uint32_t ns = 0, no = 0;
            for (PriceType x : same) ns += x > px;
            for (PriceType x : opp) no += x > px;
            const uint32_t es = std::min(ns, 16u), eo = std::min(no, 16u);
            bad += jt.aggr_since(1, Side::Buy, px, true) != es || jt.aggr_since(1, Side::Buy, px, false) != es + eo;
        }
        const JoinTape& tp = jt.tape(1, Side::Buy);
        auto top16 = [](std::vector<PriceType> xs) {                    // offsets, descending, padded
            std::vector<int32_t> o;
            for (PriceType x : xs) o.push_back(int32_t(x - 100000));
            std::sort(o.rbegin(), o.rend());
            o.resize(16, kTapeEmpty);
            return o;
        };
        const std::vector<int32_t> es = top16(same), eo = top16(opp);
        const bool exact = std::equal(es.begin(), es.end(), tp.same) && std::equal(eo.begin(), eo.end(), tp.opp);
        CHECK(bad == 0 && exact, "%d prices wrong, tape exact %d", bad, int(exact));
    }

    // The far touch from a real book: odd lots at the top of the offers are
    // skipped; the size is the total from the top through the round-lot price.
    static void far_touch_from_the_book() {
        MiniBook bk;
        g_vbook = &bk;
        struct Detach { ~Detach() { g_vbook = nullptr; } } detach;
        bk.add(1, 1, Side::Sell, 100300, 50);
        bk.add(2, 1, Side::Sell, 100400, 100);
        bk.add(3, 1, Side::Buy, 100000, 300);
        JoinTactic jt(4); jt.arm(1, true);
        Sig got;
        const PriceType prev = bk.best(1, Side::Buy);
        bk.add(4, 1, Side::Buy, 100100, 200);
        jt.on_add(1, Side::Buy, 100100, 200, prev, 9, rec(got));
        const JoinSide& b = jt.signal(1, Side::Buy);
        CHECK(prev == 100000 && got.size() == 1 && b.far_px == 100400 && b.far_sz == 150,
              "far %lld/%u", (long long)b.far_px, b.far_sz);
    }

    // prefetch(sym): no live signal, one side live, both — the lines and
    // the live bits are as they were, and counting goes on.
    static void prefetch_is_only_a_hint() {
        JoinTactic jt(8);
        for (uint16_t y = 0; y < 8; ++y) jt.arm(y, true);
        Sig got;
        jt.on_add(1, Side::Buy, 100100, 300, 100000, 5, rec(got));           // 1: the bid side live
        jt.on_add(2, Side::Buy, 100100, 300, 100000, 5, rec(got));           // 2: both sides
        jt.on_add(2, Side::Sell, 100300, 300, 100400, 5, rec(got));
        jt.on_execute(1, Side::Buy, 100100, 100);
        jt.on_execute(2, Side::Sell, 100200, 100);
        TacticShared before[8];
        for (uint16_t y = 0; y < 8; ++y) std::memcpy(static_cast<void*>(&before[y]), &g_st[y].tac, sizeof(TacticShared));
        for (int rep_ = 0; rep_ < 3; ++rep_)
            for (uint16_t y = 0; y < 8; ++y) jt.prefetch(y);
        bool same = true;
        for (uint16_t y = 0; y < 8; ++y) same = same && std::memcmp(&before[y], &g_st[y].tac, sizeof(TacticShared)) == 0;
        CHECK(got.size() == 3 && same && jt.live(1, Side::Buy) && !jt.live(1, Side::Sell) && jt.live(2, Side::Buy) &&
              jt.live(2, Side::Sell) && !jt.live(0, Side::Buy), "prefetch changed join state");
        jt.on_execute(1, Side::Buy, 100100, 100);
        CHECK(jt.signal(1, Side::Buy).n_at == 2, "n_at %u", jt.signal(1, Side::Buy).n_at);
    }

    static void run_all() {
        run("join", "signal: one round-lot order improves the bid",       signal_on_round_lot_improvement);
        run("join", "no signal: odd lot, at/behind best, empty side, ...", no_signal_cases);
        run("join", "sell side: lower offer; arm copies the round lot",    sell_side_signal_uses_the_round_lot);
        run("join", "counts executions at and through the signal price",  counts_executions_at_and_through);
        run("join", "while a signal lives, new levels are not signals",    new_levels_are_not_new_signals);
        run("join", "the kill flag ends the signal when read",             kill_ends_the_signal_when_read);
        run("join", "disarm stops new signals only",                       disarm_stops_new_signals_only);
        run("join", "tape: executions beyond any price since the signal",  tape_counts_beyond_any_price);
        run("join", "tape: the 16 most aggressive, counts exact to 16",    tape_keeps_the_most_aggressive);
        run("join", "far touch: this venue's round-lot best, from book",   far_touch_from_the_book);
        run("join", "prefetch(sym) is only a hint",                        prefetch_is_only_a_hint);
    }
};

// (a) and an order ahead growing in place: re-checked only where the venue
// keeps an order's priority when its size goes up (kSizeUp, the build option
// MD_SIZE_UP_KEEPS_PRIORITY); elsewhere a size increase is a new order at the
// back and the queue ahead of ours only shrinks. Three orders of 50 ahead of
// our 100 (150 < 300) hold. Growth behind ours, or of ours, never fires;
// with ours at 200 (K x 200 = 600), the second order ahead growing by 300
// (450 ahead) does not either; by 200 more (650) it fires (a) on that
// update with kSizeUp, and nothing without. One of OURS growing counts too,
// for our orders behind it: X and Y behind three orders of 50; X grows by 500
// and Y, now with 750 shares ahead, is decided; X is not.
template <bool kSimd>
static void queue_rechecked_on_size_up() {
    auto one = [](auto tag) {
        constexpr bool kUp = decltype(tag)::value;
        using CT = CancelTacticT<kSimd, VolSource::Nbbo, kUp>;
        using S = Suite<CT>;
        MiniBook bk;
        CancelParams p; p.queue_n = 3; p.queue_k = 3.0; p.abandon_pct = 50.0;
        CT ct(4, 16, bk.pool, p); ct.set_symbol(1, T, T);
        g_nbbo[1] = {100000, 100100};
        for (uint64_t k = 1000; k < 1003; ++k) bk.add(k, 1, Side::Buy, 99900, 50);
        typename S::Rec r;
        const auto h = S::place_cb(ct, bk, 7, 1, Side::Buy, 99900, 100, 1'000, r.cb());
        bk.add(1003, 1, Side::Buy, 99900, 100);
        ct.on_book_update(1, BookEvent::Add, Side::Buy, 99900, 1'500, 0, r.cb());
        uint8_t f = bk.grow(1003, 5000);                                      // behind ours
        ct.on_book_update(1, BookEvent::Modify, Side::Buy, 99900, 2'000, f, r.cb());
        CHECK(r.v.empty() && f == kQGrew, "kSizeUp %d: growth behind (flags %u, %zu decisions)", kUp, f, r.v.size());
        f = bk.grow(7, 100);                                                  // ours
        ct.on_book_update(1, BookEvent::Modify, Side::Buy, 99900, 3'000, f, r.cb());
        CHECK(r.v.empty() && f == (kQGrew | kQTracked), "kSizeUp %d: ours grew (flags %u)", kUp, f);
        f = bk.grow(1001, 300);                                               // ahead: 450 < 600
        ct.on_book_update(1, BookEvent::Modify, Side::Buy, 99900, 3'500, f, r.cb());
        CHECK(r.v.empty() && f == kQGrew, "kSizeUp %d: fired under K x our size", kUp);
        f = bk.grow(1001, 200);                                               // ahead: 650 >= 600
        ct.on_book_update(1, BookEvent::Modify, Side::Buy, 99900, 4'000, f, r.cb());
        CHECK(f == kQGrew && r.v.size() == (kUp ? 1u : 0u) && ct.order(h)->active == !kUp &&
              (r.v.empty() || (r.v[0].h == h && r.v[0].d == CT::Decision::CancelQueue)),
              "kSizeUp %d: growth ahead, %zu decisions", kUp, r.v.size());
        MiniBook b2;
        CT c2(4, 16, b2.pool, p); c2.set_symbol(1, T, T);
        for (uint64_t k = 1000; k < 1003; ++k) b2.add(k, 1, Side::Sell, 100300, 50);
        typename S::Rec r2;
        const auto x = S::place_cb(c2, b2, 7, 1, Side::Sell, 100300, 100, 1'000, r2.cb());
        const auto y = S::place_cb(c2, b2, 8, 1, Side::Sell, 100300, 100, 1'000, r2.cb());
        f = b2.grow(7, 500);
        c2.on_book_update(1, BookEvent::Modify, Side::Sell, 100300, 2'000, f, r2.cb());
        CHECK(f == (kQGrew | kQTracked) && r2.v.size() == (kUp ? 1u : 0u) && c2.order(x)->active &&
              c2.order(y)->active == !kUp && (r2.v.empty() || (r2.v[0].h == y && r2.v[0].d == CT::Decision::CancelQueue)),
              "kSizeUp %d: ours grew ahead of ours, %zu decisions", kUp, r2.v.size());
    };
    one(std::true_type{});
    one(std::false_type{});
}

// ---- (a), (b) and the join tactic against models written from the requirements ---------------
//
// The differential below compares two implementations of the same logic; these
// compare the logic with brute force. One symbol, three prices a side, random
// streams of market orders joining, leaving (deleted, cancelled, replaced or
// filled), cut and grown in place, and of our orders placed, cut, grown,
// filled and cancelled, with every decision acked, rejected (rearm) or left
// in flight. (c) and (d) are out of the way (no NBBO; T out of reach).
//   (a) N orders and K x our size ahead: at placement; again for our active
//       orders at a price when one of ours there is cut, or (kSizeUp) when
//       any order there grows.
//   (b) on a removal at its price that finds our order at the front, with
//       more than X% of its round-lot orders gone by cancellation.
//       Requirement (own_sets): each order of ours has its own set — the
//       market orders ahead of it holding a round lot when it was placed.
//       Design: one flag per market order. A new order's scan rewrites the
//       flags back to the nearest order of ours ahead of it and only adds
//       from there on, so the two differ only when several of ours share a
//       price and a market order ahead of the earlier one is cut or grown.
struct ModelOpt { bool own_sets, multi, cuts, growth, deep; };

template <bool kSimd, bool kUp>
struct QueueModel {
    using CT = CancelTacticT<kSimd, VolSource::Nbbo, kUp>;
    using H = typename CT::Handle;
    using D = typename CT::Decision;
    struct MOrd {
        uint64_t ref; uint32_t sz; uint32_t qh;
        bool ours = false, flag = false;
        H h = 0; bool lane = false, active = false, armed = false;
        uint32_t n0 = 0, nc = 0;
        std::vector<uint64_t> own;                         // own_sets: its round-lot orders still ahead
    };
    struct Level { QBlockList q; std::vector<MOrd> v; };   // v in time priority
    struct Got { H h; uint64_t id; D d; };

    static bool run_one(uint64_t seed, const ModelOpt& o, std::size_t& decisions, char* why, std::size_t whyn) {
        std::mt19937_64 rng(seed);
        auto U = [&](uint64_t n) { return rng() % n; };
        static constexpr uint32_t Ns[] = {0, 1, 2, 3, 5};
        static constexpr double Ks[] = {0.5, 1.0, 2.0, 3.0};
        static constexpr double Xs[] = {-1.0, 0.0, 30.0, 50.0, 75.0};
        CancelParams p;
        p.queue_n = Ns[U(5)]; p.queue_k = Ks[U(4)]; p.abandon_pct = Xs[U(5)];
        p.aggr_prints_n = 2'000'000'000; p.aggr_from_signal = false;
        const bool a_on = p.queue_n != 0, b_on = p.abandon_pct >= 0.0;
        QBlockPool pool(2048);
        CT ct(4, 64, pool, p);
        ct.set_symbol(1, T, T);
        std::map<std::pair<int, PriceType>, Level> lv;
        std::vector<Got> got;
        auto cb = [&](H h, uint64_t id, D d) { got.push_back({h, id, d}); };
        std::vector<std::pair<uint64_t, D>> exp;
        uint64_t next_ref = 1;
        FeedTimeType ts = 1'000'000;
        int lanes[2] = {0, 0};
        auto cond_a = [&](const Level& L, std::size_t i) {
            uint64_t sz = 0;
            for (std::size_t k = 0; k < i; ++k) sz += L.v[k].sz;
            return a_on && i >= p.queue_n && static_cast<double>(sz) >= p.queue_k * static_cast<double>(L.v[i].sz);
        };
        auto ratio = [&](const MOrd& m) {
            return m.n0 != 0 && 100.0 * static_cast<double>(m.nc) > p.abandon_pct * static_cast<double>(m.n0);
        };
        auto recheck_a = [&](Level& L) {
            for (std::size_t i = 0; i < L.v.size(); ++i) {
                MOrd& m = L.v[i];
                if (m.ours && m.lane && m.active && cond_a(L, i)) {
                    exp.push_back({m.ref, D::CancelQueue});
                    m.active = m.armed = false;
                }
            }
        };
        for (int step = 0; step < 400; ++step) {
            ts += 1000;
            const int side = static_cast<int>(U(2));
            const PriceType px = 100000 + (side ? 300 : -300) +
                                 static_cast<PriceType>(o.deep ? 0 : U(3)) * T * (side ? 1 : -1);
            const Side sd = static_cast<Side>(side);
            Level& L = lv[{side, px}];
            got.clear(); exp.clear();
            unsigned r = static_cast<unsigned>(U(100));
            if (o.deep && r >= 26 && r < 62 && U(100) < 55) r = 0;            // deep: levels spill into chains
            const char* what;
            if (r < 26) {                                                     // a market order joins
                what = "market order joins";
                MOrd m; m.ref = next_ref++;
                m.sz = U(100) < 40 ? static_cast<uint32_t>(10 + U(85)) : static_cast<uint32_t>(100 + U(300));
                m.qh = L.q.push(pool, m.ref, m.sz);
                L.v.push_back(m);
                ct.on_book_update(1, BookEvent::Add, sd, px, ts, 0, cb);
            } else if (r < 62 && !L.v.empty()) {                              // an order leaves
                const std::size_t i = U(L.v.size());
                static constexpr BookEvent kEv[3] = {BookEvent::Delete, BookEvent::Cancel, BookEvent::Replace};
                const BookEvent ev = U(100) < 45 ? BookEvent::Execute : kEv[U(3)];
                what = L.v[i].ours ? "our order leaves" : "market order leaves";
                const MOrd R = L.v[i];
                for (std::size_t k = i + 1; k < L.v.size(); ++k) {            // ours behind it, armed
                    MOrd& m = L.v[k];
                    if (!(m.ours && m.lane && m.armed)) continue;
                    if (o.own_sets) {
                        const auto it = std::find(m.own.begin(), m.own.end(), R.ref);
                        if (it != m.own.end()) { if (ev != BookEvent::Execute) ++m.nc; m.own.erase(it); }
                    } else if (R.flag && !R.ours && ev != BookEvent::Execute && m.nc < m.n0) {
                        ++m.nc;
                    }
                }
                if (R.ours && R.lane) --lanes[side];
                L.v.erase(L.v.begin() + static_cast<std::ptrdiff_t>(i));
                if (!L.v.empty()) {
                    MOrd& m = L.v[0];                                         // (b): armed, at the front
                    if (m.ours && m.lane && m.armed && ratio(m)) {
                        exp.push_back({m.ref, D::CancelAbandoned});
                        m.active = m.armed = false;
                    }
                }
                const uint8_t f = L.q.remove(pool, R.qh);
                if (f & kQTracked) ct.on_book_remove(1, sd, R.qh);
                ct.on_book_update(1, ev, sd, px, ts, f, cb);
            } else if (r < 70 && !L.v.empty()) {                              // an order is cut in place
                MOrd& m = L.v[U(L.v.size())];
                if (m.sz < 2 || (!m.ours && !o.cuts)) continue;
                what = m.ours ? "our order cut" : "market order cut";
                const uint32_t d = static_cast<uint32_t>(1 + U(m.sz - 1));
                m.sz -= d;
                const uint8_t f = L.q.reduce(pool, m.qh, d);
                if (m.ours) recheck_a(L);
                ct.on_book_update(1, U(2) ? BookEvent::Execute : BookEvent::Cancel, sd, px, ts, f, cb);
            } else if (r < 78 && !L.v.empty() && o.growth) {                  // an order grows in place
                MOrd& m = L.v[U(L.v.size())];
                what = m.ours ? "our order grows" : "market order grows";
                const uint32_t d = static_cast<uint32_t>(1 + U(300));
                m.sz += d;
                const uint8_t f = L.q.grow(pool, m.qh, d);
                if (kUp) recheck_a(L);
                ct.on_book_update(1, BookEvent::Modify, sd, px, ts, f, cb);
            } else if (r < 94 && lanes[side] < 8) {                           // one of ours joins
                bool have = false;
                for (const MOrd& m : L.v) have = have || m.ours;
                if (have && !o.multi) continue;
                what = "our order placed";
                MOrd m; m.ref = next_ref++; m.ours = true;
                m.sz = static_cast<uint32_t>(50 * (1 + U(4)));
                if (b_on) {
                    bool exact = true;                                        // nothing of ours met yet
                    for (std::size_t k = L.v.size(); k-- > 0;) {              // from the nearest ahead
                        MOrd& x = L.v[k];
                        if (x.ours) { exact = false; continue; }
                        if (x.sz >= 100) m.own.push_back(x.ref);
                        x.flag = x.sz >= 100 || (!exact && x.flag);
                        m.n0 += x.flag;
                    }
                    if (o.own_sets) m.n0 = static_cast<uint32_t>(m.own.size());
                }
                m.qh = L.q.push(pool, m.ref, m.sz);
                L.v.push_back(m);
                MOrd& me = L.v.back();
                me.lane = true; ++lanes[side];
                if (cond_a(L, L.v.size() - 1)) exp.push_back({me.ref, D::CancelQueue});
                else { me.active = true; me.armed = me.n0 != 0; }
                me.h = ct.on_place(me.ref, 1, sd, px, ts, me.qh, cb);
            } else {
                continue;
            }
            std::vector<std::pair<uint64_t, D>> g;
            for (const Got& x : got) g.push_back({x.id, x.d});
            std::sort(g.begin(), g.end()); std::sort(exp.begin(), exp.end());
            decisions += g.size();
            bool ok = g == exp;
            for (const Got& x : got) {                                        // the host's answer
                MOrd* m = nullptr; int ms = 0;
                for (auto& [k, lev] : lv)
                    for (MOrd& y : lev.v) if (y.ref == x.id) { m = &y; ms = k.first; }
                if (!m) { ok = false; break; }
                const unsigned a = static_cast<unsigned>(U(100));
                if (a < 30) { ok = ok && ct.on_done(x.h); m->lane = m->active = m->armed = false; --lanes[ms]; }
                else if (a < 60) { ok = ok && ct.rearm(x.h); m->active = true; m->armed = m->n0 != 0; }
            }
            for (auto& [k, lev] : lv)
                for (const MOrd& m : lev.v) {
                    if (!(m.ours && m.lane)) continue;
                    const auto v = ct.order(m.h);
                    ok = ok && v && v->active == m.active &&
                         (o.own_sets || (v->ahead_lot_n == m.n0 && v->ahead_cancelled == m.nc));
                }
            if (!ok) {
                std::snprintf(why, whyn, "seed %llu step %d (%s; N=%u K=%.1f X=%.0f): tactic decided %zu, model %zu",
                              (unsigned long long)seed, step, what, p.queue_n, p.queue_k, p.abandon_pct, g.size(),
                              exp.size());
                return false;
            }
        }
        return true;
    }

    static void campaign(const char* name, const ModelOpt& o, int runs = 400) {
        int bad = 0; char first[200] = ""; char why[200]; std::size_t dec = 0;
        for (int i = 0; i < runs; ++i)
            if (!run_one(1000 + static_cast<uint64_t>(i), o, dec, why, sizeof why) && !bad++)
                std::snprintf(first, sizeof first, "%s", why);
        CHECK(bad == 0 && dec > 1000, "%s: %d of %d streams disagree (%zu decisions); %s", name, bad, runs, dec, first);
    }
    static void design()   {                                                 // the delivered rule, exactly
        campaign("flags as designed", {false, true, true, true, false});
        campaign("flags as designed, deep levels", {false, true, true, true, true});
    }
    static void required() {                                                 // where design == requirement
        campaign("own sets, one of ours per price", {true, false, true, true, false});
        campaign("own sets, one of ours per price, deep levels", {true, false, true, true, true});
        campaign("own sets, several per price, market orders neither cut nor grown", {true, true, false, false, false});
    }
};

// The join tactic: signals, lifetime (kill flag read on executions, improving
// adds and poll()), counts, and the (d) start for any price, against lists of
// the executions since each signal.
static void join_model() {
    int bad = 0, sigs = 0;
    for (int run = 0; run < 400 && !bad; ++run) {
        std::mt19937_64 rng(77 + static_cast<uint64_t>(run));
        auto U = [&](uint64_t n) { return rng() % n; };
        g_st[1].tac = TacticShared{};
        JoinTactic jt(4);
        struct M { bool live = false, kill = false; PriceType px = 0; uint32_t seq = 0, n_at = 0, same_n = 0, opp_n = 0;
                   std::vector<PriceType> same, opp; } m[2];
        bool armed = false;
        int sig = 0, want = 0;
        auto on_sig = [&](uint16_t, Side, const JoinSide&) { ++sig; };
        auto end = [](M& x) { x.live = x.kill = false; x.px = 0; };
        const PriceType B = 100000;
        for (int step = 0; step < 600 && !bad; ++step) {
            const unsigned r = static_cast<unsigned>(U(100));
            const int s = static_cast<int>(U(2));
            if (r < 6) { armed = U(3) != 0; jt.arm(1, armed); }
            else if (r < 30) {                                                 // an Add
                const PriceType prev = U(12) == 0 ? 0 : B + (PriceType(U(9)) - 4) * T;
                const PriceType px = prev + (PriceType(U(5)) - 2) * T * (s ? -1 : 1);
                const uint32_t sz = U(3) ? 100 + static_cast<uint32_t>(U(300)) : 1 + static_cast<uint32_t>(U(99));
                M& x = m[s];
                if (armed && (s ? px < prev : px > prev) && prev != 0) {
                    bool go = true;
                    if (x.live) { if (x.kill) end(x); else go = false; }
                    if (go && sz >= 100) {
                        x.live = true; x.kill = false; x.px = px; ++x.seq; x.n_at = x.same_n = x.opp_n = 0;
                        x.same.clear(); x.opp.clear(); ++want;
                    }
                }
                jt.on_add(1, static_cast<Side>(s), px, sz, prev, 1000 + static_cast<FeedTimeType>(step), on_sig);
            } else if (r < 80) {                                               // an execution
                const PriceType px = B + (PriceType(U(21)) - 10) * T;
                for (int k = 0; k < 2; ++k) {
                    M& x = m[k];
                    if (!x.live) continue;
                    if (x.kill) { end(x); continue; }
                    (s == k ? x.same : x.opp).push_back(px);
                    if (px == x.px) ++x.n_at;
                    else if (k ? px < x.px : px > x.px) ++(s == k ? x.same_n : x.opp_n);
                }
                jt.on_execute(1, static_cast<Side>(s), px, 10);
            } else if (r < 88) { m[s].kill = true; JoinTactic::kill(1, static_cast<Side>(s)); }
            else if (r < 92) { for (M& x : m) if (x.live && x.kill) end(x); jt.poll(1); }
            for (int k = 0; k < 2 && !bad; ++k) {
                const M& x = m[k];
                const JoinSide& js = jt.signal(1, static_cast<Side>(k));
                bool ok = sig == want && jt.live(1, static_cast<Side>(k)) == x.live && js.px == x.px &&
                          js.seq == (x.seq & 0xFFFFu);
                if (x.live) ok = ok && js.n_at == x.n_at && js.n_thru_same == x.same_n && js.n_thru_opp == x.opp_n;
                for (int q = 0; q < 4 && ok; ++q) {
                    const PriceType px = B + (PriceType(U(25)) - 12) * T;
                    auto beyond = [&](const std::vector<PriceType>& v) {
                        uint32_t n = 0;
                        for (PriceType e : v) n += k ? e < px : e > px;
                        return std::min(n, 16u);
                    };
                    const bool on = x.live && !x.kill;
                    ok = jt.aggr_since(1, static_cast<Side>(k), px, true) == (on ? beyond(x.same) : 0u) &&
                         jt.aggr_since(1, static_cast<Side>(k), px, false) == (on ? beyond(x.same) + beyond(x.opp) : 0u);
                }
                if (!ok) { ++bad; CHECK(false, "run %d step %d side %d (signals %d, model %d)", run, step, k, sig, want); }
            }
        }
        sigs += sig;
    }
    CHECK(bad == 0 && sigs > 2000, "%d disagreements, %d signals", bad, sigs);
}

// ---- randomized differential: SIMD vs scalar reference --------------------------------------

struct TraceRec {
    uint64_t ev, h, id; int d;
    bool operator==(const TraceRec&) const = default;
};
struct Trace {
    std::vector<TraceRec> recs;
    std::vector<uint64_t> state;
    std::size_t fires[5] = {0, 0, 0, 0, 0}, events = 0, nested = 0, book_removes = 0, gated = 0, ahead_gone = 0;
    std::size_t signals = 0, joined = 0, seeded = 0, requeued = 0;   // (d) seeded; (a) decided after placement
    std::size_t grown = 0;                                           // (a) decided on an order ahead growing
};

template <class CT>
static Trace random_stream(uint64_t seed, const CancelParams& p) {
    using H = typename CT::Handle;
    using D = typename CT::Decision;
    constexpr unsigned NS = 6;
    constexpr uint64_t kOurs = 1ull << 40;                           // our refs: kOurs + n
    static constexpr BookEvent kNonExec[5] = {BookEvent::Add, BookEvent::Delete, BookEvent::Modify,
                                              BookEvent::Cancel, BookEvent::Replace};
    MiniBook bk(16384);
    g_vbook = &bk;                                                   // venue round-lot BBO (VolSource::Venue)
    struct Detach { ~Detach() { g_vbook = nullptr; } } detach;
    for (auto& st : g_st) { st.tac = TacticShared{}; st.round_lot = 100; }
    CT ct(NS, 64, bk.pool, p);
    JoinTactic jt(NS);                                               // join signals feed (d) via SymbolState
    for (unsigned s = 0; s < NS; ++s) jt.arm(static_cast<uint16_t>(s), true);
    for (unsigned s = 0; s < NS; ++s) ct.set_symbol(static_cast<uint16_t>(s), T, PriceType(50 + 50 * s));
    std::mt19937_64 rng(seed);
    auto U = [&](uint64_t n) { return rng() % n; };
    struct SymSim { PriceType bid = 100000, ask = 100100; };
    SymSim sim[NS];
    for (unsigned s = 0; s < NS; ++s) g_nbbo[s] = {sim[s].bid, sim[s].ask};
    std::vector<uint64_t> venue, ours;                               // refs in the book
    Trace tr;
    uint64_t ev = 0, next_venue = 1, next_ours = 1;
    FeedTimeType clock = 1'000'000'000;

    // Remove an order from the book the way the host would: queue first, a
    // tracked order released from the tactic, then the book update.
    auto book_remove = [&](uint64_t ref, BookEvent kind, auto& cb) {
        MiniBook::Ord o;
        const uint32_t gone_sz = QBlockList::size_of(bk.pool, bk.ord.at(ref).qh);
        const uint8_t f = bk.remove(ref, &o);
        if (kind == BookEvent::Execute) jt.on_execute(o.sym, o.sd, o.px, gone_sz);
        if (f & kQTracked) { ++tr.book_removes; ct.on_book_remove(o.sym, o.sd, o.qh); }
        if (f & kQAhead) ++tr.ahead_gone;
        ct.on_book_update(o.sym, kind, o.sd, o.px, clock, f, cb);
    };
    bool placing = false, growing = false;
    auto place_ours = [&](uint16_t sy, Side sd, PriceType px, auto& cbk) {
        const uint64_t ref = kOurs + next_ours++;
        const uint32_t qh = bk.add(ref, sy, sd, px, 100);
        if (qh == QBlockList::kBadHandle) return;
        ours.push_back(ref);
        placing = true;
        const H h = ct.on_place(ref, sy, sd, px, clock, qh, cbk);          // capacity full: in the book, untracked
        placing = false;
        if (const auto v = ct.order(h); v && v->n_aggr_prints) ++tr.seeded;
    };
    struct Join { uint16_t sy; Side sd; PriceType px; uint64_t at; };
    std::vector<Join> joins;                                         // signals we will join, a few events on,
    struct Kill { uint16_t sy; Side sd; uint64_t at; };              //   at its price or up to 2 ticks either way
    std::vector<Kill> kills;                                         // the strategy ends each signal later
    auto on_signal = [&](uint16_t sy, Side sd, const JoinSide& js) {
        ++tr.signals;
        if (U(100) < 60) joins.push_back({sy, sd, js.px + (PriceType(U(5)) - 2) * T, ev + 1 + U(40)});
        kills.push_back({sy, sd, ev + 1 + U(3000)});
    };

    std::vector<uint64_t> landing;                                   // our cancels the feed will delete
    struct Cb {
        CT& ct; std::mt19937_64& rng; Trace& tr; uint64_t& ev; FeedTimeType& clock; SymSim* sim;
        std::vector<uint64_t>& landing; const bool& placing; const bool& growing; int depth = 0;
        void operator()(H h, uint64_t id, D d) {
            tr.recs.push_back({ev, h, id, static_cast<int>(d)});
            tr.fires[static_cast<int>(d)]++;
            if (d == D::CancelQueue && !placing) ++tr.requeued;
            if (d == D::CancelQueue && growing && depth == 0) ++tr.grown;
            const unsigned r = static_cast<unsigned>(rng() % 100);
            if (r < 25) { ct.on_done(h); landing.push_back(id); }      // ack before the feed's delete
            else if (r < 45) ct.rearm(h);                               // cancel rejected
            else landing.push_back(id);                                 // the feed's delete releases it
            if (depth == 0 && rng() % 100 < 5) {                        // nested book update
                ++depth; ++tr.nested;
                const uint16_t sy = static_cast<uint16_t>(rng() % NS);
                SymSim& m = sim[sy];
                if (rng() % 2) {
                    const PriceType st = (rng() & 1) ? T : -T;
                    m.bid += st;
                    if (m.ask <= m.bid) m.ask = m.bid + T;
                    g_nbbo[sy] = {m.bid, m.ask};
                }
                if (rng() % 2) {
                    const Side rest = (rng() & 1) ? Side::Buy : Side::Sell;
                    ct.on_book_update(sy, BookEvent::Execute, rest, rest == Side::Buy ? m.bid : m.ask, clock, 0, *this);
                } else {
                    ct.on_book_update(sy, BookEvent::Modify, Side::Buy, 0, clock, 0, *this);
                }
                --depth;
            }
        }
    } cb{ct, rng, tr, ev, clock, sim, landing, placing, growing};

    constexpr uint64_t kEvents = 200'000;
    for (ev = 1; ev <= kEvents; ++ev) {
        const uint16_t sy = static_cast<uint16_t>(U(NS));
        SymSim& m = sim[sy];
        ct.prefetch(sy); jt.prefetch(sy);                                 // decode stage: hints only
        if (U(100) >= 15) clock += U(400'000);                            // 15%: same timestamp as the last
        for (std::size_t i = 0; i < kills.size();)                        // the strategy sets a kill flag; the
            if (kills[i].at <= ev) {                                      //   join tactic reads it later
                JoinTactic::kill(kills[i].sy, kills[i].sd);
                kills[i] = kills.back(); kills.pop_back();
            } else ++i;
        if (U(100) < 35) {                                                // the NBBO moved
            const unsigned q = static_cast<unsigned>(U(100));
            if (q < 30)      { const PriceType st = U(2) ? T : -T; m.bid += st * PriceType(1 + U(2)); }
            else if (q < 60) { const PriceType st = U(2) ? T : -T; m.ask += st * PriceType(1 + U(2)); }
            else if (q < 70) { m.bid += T; m.ask += T; }
            else if (q < 80) { m.bid -= T; m.ask -= T; }
            if (m.ask <= m.bid) m.ask = m.bid + T * PriceType(1 + U(3));
            PriceType b = m.bid, a = m.ask;
            const unsigned deg = static_cast<unsigned>(U(1000));
            if (deg < 5) b = 0; else if (deg < 10) a = 0;                 // one-sided
            else if (deg < 13) a = b - T;                                 // crossed
            else if (deg < 30) a = b;                                     // locked
            g_nbbo[sy] = {b, a};
        }
        if (!landing.empty() && U(100) < 30) {                            // a cancel lands: the feed deletes
            const uint64_t ref = landing.front();                         //   our order (if still in the book)
            landing.erase(landing.begin());
            const auto it = std::find(ours.begin(), ours.end(), ref);
            if (it != ours.end()) {
                *it = ours.back(); ours.pop_back();
                book_remove(ref, BookEvent::Delete, cb);
                continue;
            }
        }
        if (!joins.empty() && joins.front().at <= ev) {                   // join a signal's level
            const Join j = joins.front();
            joins.erase(joins.begin());
            place_ours(j.sy, j.sd, j.px, cb);
            ++tr.joined;
            continue;
        }
        const unsigned kind = static_cast<unsigned>(U(100));
        if (kind < 24) {                                                  // venue add near the touch
            const Side sd = U(2) ? Side::Buy : Side::Sell;
            const PriceType off = U(100) < 12 ? -T : PriceType(U(6)) * T; //   sometimes improving
            const PriceType px = sd == Side::Buy ? m.bid - off : m.ask + off;
            const uint32_t sz = static_cast<uint32_t>(1 + U(U(100) < 40 ? 80 : 500));   // 40% small
            const PriceType prev_best = bk.best(sy, sd);
            if (px > 0 && bk.add(next_venue, sy, sd, px, sz) != QBlockList::kBadHandle) {
                venue.push_back(next_venue++);
                jt.on_add(sy, sd, px, sz, prev_best, clock, on_signal);
                ct.on_book_update(sy, BookEvent::Add, sd, px, clock, 0, cb);
            }
        } else if (kind < 46 && !venue.empty()) {                         // venue delete / execution
            const std::size_t i = U(venue.size());
            const uint64_t ref = venue[i];
            venue[i] = venue.back(); venue.pop_back();
            const BookEvent k = U(2) ? BookEvent::Execute : BookEvent::Delete;
            book_remove(ref, k, cb);
        } else if (kind < 50 && !venue.empty()) {                         // venue order cut or grown in place
            uint64_t ref = venue[U(venue.size())];                        //   (a grow re-checks (a) only if
            if (U(2) && !ours.empty()) {                                  //   the venue keeps its priority);
                const auto& w = bk.ord.at(ours[U(ours.size())]);          //   half the time one at a level
                std::size_t n = 0;                                        //   where one of ours rests
                for (uint64_t v : venue) {
                    const auto& x = bk.ord.at(v);
                    if (x.sym == w.sym && x.sd == w.sd && x.px == w.px && U(++n) == 0) ref = v;
                }
            }
            const auto& o = bk.ord.at(ref);
            const uint32_t sz = QBlockList::size_of(bk.pool, o.qh);
            uint8_t f = 0;
            if (U(2) && sz > 1) f = bk.reduce(ref, static_cast<uint32_t>(1 + U(sz - 1)));
            else f = bk.grow(ref, static_cast<uint32_t>(1 + U(400)));
            growing = (f & kQGrew) != 0;
            ct.on_book_update(o.sym, BookEvent::Modify, o.sd, o.px, clock, f, cb);
            growing = false;
        } else if (kind < 58) {                                           // other updates (no book change)
            FeedTimeType ts = clock;
            if (U(100) < 3) ts = clock - U(50'000);                       // early stamp
            const BookEvent k = kNonExec[U(5)];
            ct.on_book_update(sy, k, U(2) ? Side::Buy : Side::Sell, m.bid, ts, 0, cb);
        } else if (kind < 75) {                                           // sweep: 1-4 executions, one ts
            const Side rest = U(2) ? Side::Buy : Side::Sell;
            const unsigned n = U(100) < 15 ? 1 + static_cast<unsigned>(U(4)) : 1;
            const bool off_touch = U(100) < 10;
            FeedTimeType ts = clock;
            if (U(100) < 2) ts = clock - U(50'000);
            for (unsigned k = 0; k < n; ++k) {
                PriceType px = rest == Side::Buy ? m.bid - PriceType(k) * T : m.ask + PriceType(k) * T;
                if (off_touch) px += (PriceType(U(7)) - 3) * T;
                jt.on_execute(sy, rest, px, static_cast<uint32_t>(1 + U(300)));
                ct.on_book_update(sy, BookEvent::Execute, rest, px, ts, 0, cb);
            }
        } else if (kind < 83) {                                           // our order enters the book
            const Side sd = U(2) ? Side::Buy : Side::Sell;
            const PriceType off = PriceType(U(9)) * T;
            const PriceType px = sd == Side::Buy ? m.bid - off : m.ask + off;
            if (px > 0) place_ours(sy, sd, px, cb);
        } else if (kind < 90 && !ours.empty()) {                          // our order leaves the book
            const std::size_t i = U(ours.size());
            const uint64_t ref = ours[i];
            ours[i] = ours.back(); ours.pop_back();
            book_remove(ref, U(2) ? BookEvent::Execute : BookEvent::Delete, cb);
        } else if (kind < 95 && !ours.empty()) {                          // ours partially filled or cut
            const uint64_t ref = ours[U(ours.size())];                    //   (re-checks (a)); grown 1 in 5
            const auto& o = bk.ord.at(ref);
            const uint32_t sz = QBlockList::size_of(bk.pool, o.qh);
            const unsigned how = static_cast<unsigned>(U(5));
            BookEvent k = BookEvent::Execute;
            uint8_t f = 0;
            if (how == 4) { f = bk.grow(ref, static_cast<uint32_t>(1 + U(200))); k = BookEvent::Modify; }
            else if (sz > 1) {
                const uint32_t dlt = static_cast<uint32_t>(1 + U(sz - 1));
                f = bk.reduce(ref, dlt);
                if (how < 2) jt.on_execute(o.sym, o.sd, o.px, dlt);
                else k = BookEvent::Cancel;
            }
            ct.on_book_update(o.sym, k, o.sd, o.px, clock, f, cb);
        } else if (kind < 97) {                                           // the join tactic polls; now and
            jt.poll(sy);                                                  //   then a kill flag set early, or
            if (U(4) == 0) JoinTactic::kill(sy, U(2) ? Side::Buy : Side::Sell);   //   with no live signal
        } else {                                                          // pin toggles
            ct.set_tracking(sy, U(2) != 0);
        }
    }
    auto bits = [](double x) { uint64_t u; std::memcpy(&u, &x, 8); return u; };
    for (unsigned s = 0; s < NS; ++s) {
        const uint16_t sy = static_cast<uint16_t>(s);
        tr.state.push_back(static_cast<uint64_t>(ct.threshold(sy)));
        tr.state.push_back(bits(ct.sigma_rate(sy)));
        tr.state.push_back(bits(ct.trade_gap_ns(sy)));
        tr.state.push_back(ct.sym_time(sy));
        tr.state.push_back(ct.tracking(sy));
    }
    for (uint64_t ref : ours) {
        const auto& o = bk.ord.at(ref);
        const H h = ct.find(o.sym, o.sd, o.qh);
        const auto v = ct.order(h);
        tr.state.push_back(v ? (v->id ^ (uint64_t(v->n_aggr_prints) << 40) ^ (uint64_t(v->active) << 63) ^
                                (uint64_t(v->queue_ahead) << 48) ^ v->queue_ahead_sz ^
                                (uint64_t(v->ahead_lot_n) << 20) ^ (uint64_t(v->ahead_cancelled) << 30)) : ~0ull);
        tr.state.push_back(static_cast<uint64_t>(ct.evaluate(h)));
    }
    tr.state.push_back(ct.live_orders());
    tr.state.push_back(ct.place_failures());
    tr.events = kEvents;
    return tr;
}

template <VolSource kV, bool kUp = false>
static void differential_simd_vs_reference() {
#if !(defined(__AVX512VL__) && defined(__AVX512DQ__))
    std::printf("      (no AVX-512VL/DQ in this build: skipped)\n");
#else
    CancelParams p1;
    p1.aggr_prints_n = 8; p1.min_ticks = 2; p1.far_touch_min_stable_ns = 50'000;
    p1.vol_halflife_ns = 2'000'000'000; p1.gap_mult = 2.0;
    p1.queue_n = 4; p1.queue_k = 3.0; p1.abandon_pct = 50.0;
    CancelParams p2 = p1;
    p2.queue_n = 2; p2.queue_k = 1.0; p2.abandon_pct = 0.0; p2.aggr_from_signal = false;
    p2.passive_side_only = false; p2.aggr_gate_n = 2; p2.aggr_gate_sz = 150;
    p2.trade_coalesce_ns = 20'000; p2.place_spread_mult = 0.5; p2.hist_spread_mult = 2.0;
    p2.min_ticks = 1; p2.aggr_prints_n = 30; p2.gap_mult = 1.0;
    CancelParams p3 = p1;                                            // (a) off: room for (b) and (d)
    p3.queue_n = 0; p3.abandon_pct = 30.0; p3.aggr_prints_n = 5;
    CancelParams p4 = p1;                                            // (c) alone: orders stay live, sides
    p4.queue_n = 0; p4.abandon_pct = -1.0; p4.aggr_prints_n = 1'000'000;   //   fill up, the SIMD (c)
    p4.min_ticks = 1; p4.hist_spread_mult = 0.5; p4.gap_mult = 4.0;          //   kernel decides
    std::size_t total = 0, fq = 0, fab = 0, fa = 0, fb = 0, nested = 0, removes = 0, ahead = 0, sig = 0, joined = 0,
                seeded = 0, requeued = 0, grown = 0;
    for (const CancelParams* p : {&p1, &p2, &p3, &p4})
        for (uint64_t seed : {1ull, 7ull, 12345ull}) {
            const Trace v = random_stream<CancelTacticT<true, kV, kUp>>(seed, *p);
            const Trace r = random_stream<CancelTacticT<false, kV, kUp>>(seed, *p);
            std::size_t i = 0;
            while (i < v.recs.size() && i < r.recs.size() && v.recs[i] == r.recs[i]) ++i;
            CHECK(v.recs.size() == r.recs.size() && i == v.recs.size(),
                  "seed %llu: decisions diverge at #%zu (simd %zu, ref %zu)",
                  (unsigned long long)seed, i, v.recs.size(), r.recs.size());
            CHECK(v.state == r.state, "seed %llu: final state differs", (unsigned long long)seed);
            total += v.events; nested += v.nested; removes += v.book_removes; ahead += v.ahead_gone;
            fq += v.fires[1]; fab += v.fires[2]; fa += v.fires[3]; fb += v.fires[4];
            sig += v.signals; joined += v.joined; seeded += v.seeded; requeued += v.requeued; grown += v.grown;
        }
    CHECK(kUp ? grown > 100 : grown == 0, "(a) on growth ahead: %zu decisions", grown);
    CHECK(fq > 100 && fab > 50 && fa > 300 && fb > 1000 && removes > 1000 && joined > 500 && seeded > 1000 &&
          requeued > 1000,
          "stream too tame: %zu queue (%zu after placement) / %zu abandoned / %zu depth / %zu outcompeted / "
          "%zu removes / %zu joins / %zu seeded", fq, requeued, fab, fa, fb, removes, joined, seeded);
    std::printf("      %zu updates, decisions: %zu queue (%zu after placement, %zu on growth ahead) +\n"
                "      %zu abandoned + %zu depth + %zu outcompeted; %zu join signals (%zu joined),\n"
                "      %zu orders seeded from one, %zu counted-ahead orders left, %zu tracked orders\n"
                "      left the book, %zu nested updates: identical\n",
                total, fq, requeued, grown, fab, fa, fb, sig, joined, seeded, ahead, removes, nested);
#endif
}

int main() {
    std::printf("CancelTactic regression tests (SIMD kernels %s in this build)\n",
                kCancelTacticSimd ? "available" : "not available");
    run("kernel", "exp2_neg / decay accuracy vs std::exp2", exp2_accuracy);
    run("qblock", "QBlockList layout: 32-byte blocks, headers in one line", qblocklist_layout);
    run("qblock", "QBlockList vs reference model (flags, scans, grow, ahead_n)", qblocklist_model);
    JoinSuite::run_all();
#if defined(__AVX512VL__) && defined(__AVX512DQ__)
    Suite<CancelTacticT<true, VolSource::Nbbo>>::run_all("simd");
    VenueSuite<true>::run_all("simd");
#endif
    Suite<CancelTacticT<false, VolSource::Nbbo>>::run_all("ref ");
    VenueSuite<false>::run_all("ref ");
#if defined(__AVX512VL__) && defined(__AVX512DQ__)
    run("simd", "10 (a) re-checked on a size-up only where it keeps priority", queue_rechecked_on_size_up<true>);
#endif
    run("ref ", "10 (a) re-checked on a size-up only where it keeps priority", queue_rechecked_on_size_up<false>);
    run("model", "join tactic == model: signals, kill, counts, (d) start", join_model);
#if defined(__AVX512VL__) && defined(__AVX512DQ__)
    run("model", "simd (a)/(b) == model of the design",                 QueueModel<true, false>::design);
    run("model", "simd (a)/(b) == model of the design, size-up",        QueueModel<true, true>::design);
    run("model", "simd (a)/(b) == the requirement, per-order sets",     QueueModel<true, false>::required);
    run("model", "simd (a)/(b) == the requirement, size-up",            QueueModel<true, true>::required);
#endif
    run("model", "ref  (a)/(b) == model of the design",                 QueueModel<false, false>::design);
    run("model", "ref  (a)/(b) == model of the design, size-up",        QueueModel<false, true>::design);
    run("model", "ref  (a)/(b) == the requirement, per-order sets",     QueueModel<false, false>::required);
    run("model", "ref  (a)/(b) == the requirement, size-up",            QueueModel<false, true>::required);
    run("diff", "randomized stream, NBBO vol: SIMD == reference, bit for bit",
        differential_simd_vs_reference<VolSource::Nbbo>);
    run("diff", "randomized stream, venue round-lot vol: SIMD == reference, bit for bit",
        differential_simd_vs_reference<VolSource::Venue>);
    run("diff", "randomized stream, size-up keeps priority: SIMD == reference, bit for bit",
        differential_simd_vs_reference<VolSource::Nbbo, true>);
    std::printf("%d/%d checks passed\n", g_checks - g_fail, g_checks);
    return g_fail ? 1 : 0;
}
