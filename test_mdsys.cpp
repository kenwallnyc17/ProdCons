#include "mdsys.hpp"
#include <chrono>
#include <cstdio>
#include <map>
#include <random>
#include <unordered_map>
#include <vector>

using namespace md;

static int failures = 0;
#define CHECK(c) do { if(!(c)){ std::printf("FAIL %s:%d  %s\n",__FILE__,__LINE__,#c); ++failures; } } while(0)

struct RefOrder { uint16_t sym; uint8_t sd; PriceType prc; SizeType sz; };
struct RefLevel { uint64_t agg = 0; uint32_t n = 0; };
using RefBook = std::map<PriceType, RefLevel>;   // ascending

static void verify_books(const MarketDataSystem& mds, uint16_t sy,
                         const RefBook rb[2], uint32_t lot) {
    for (unsigned sd = 0; sd < 2; ++sd) {
        const bool bid = (sd == 0);
        // full traversal must match reference exactly, in order
        std::vector<std::pair<PriceType, std::pair<uint64_t, uint32_t>>> got, want;
        mds.for_each_level(sy, static_cast<Side>(sd),
            [&](PriceType p, const PrcLevelGeneric& L) {
                got.push_back({p, {L.agg_sz, L.num_orders}});
                return true;
            });
        if (bid) for (auto it = rb[sd].rbegin(); it != rb[sd].rend(); ++it)
            want.push_back({it->first, {it->second.agg, it->second.n}});
        else     for (auto it = rb[sd].begin();  it != rb[sd].end();  ++it)
            want.push_back({it->first, {it->second.agg, it->second.n}});
        CHECK(got == want);
        // best
        auto b = mds.best(sy, static_cast<Side>(sd));
        if (rb[sd].empty()) CHECK(!b);
        else CHECK(b && *b == (bid ? rb[sd].rbegin()->first : rb[sd].begin()->first));
        // roundlot
        auto r = mds.roundlot_best(sy, static_cast<Side>(sd));
        uint64_t cum = 0; uint32_t used = 0; PriceType bp = 0; bool found = false;
        for (auto& [p, s] : want) { cum += s.first; ++used;
            if (cum >= lot) { bp = p; found = true; break; } }
        CHECK(r.has_value() == found);
        if (r && found) { CHECK(r->price == bp && r->cum_size == cum
                                && r->levels_used == used); }
    }
}

int main() {
    // =====================================================================
    // 1) multi-symbol churn vs reference (adds, partial/full reduce, delete,
    //    replace), periodic full-book + best + roundlot verification
    // =====================================================================
    {
        const uint16_t NSYM = 6;
        MarketDataSystem mds(NSYM, /*peak_orders*/50000, /*npages*/512,
                             /*swan*/4096);
        for (uint16_t s = 0; s < NSYM; ++s)
            mds.configure_symbol(s, /*tick*/100, /*lot*/100, /*window*/8192);

        std::unordered_map<OrderRef, RefOrder> rord;
        RefBook rbook[NSYM][2];
        std::mt19937_64 rng(20260718);
        OrderRef next_ref = 1;
        std::vector<OrderRef> live;
        uint64_t t = 0, u = 0;

        auto ref_reduce = [&](OrderRef ref, SizeType cut, bool gone) {
            auto& o = rord[ref];
            auto& L = rbook[o.sym][o.sd][o.prc];
            L.agg -= cut; o.sz -= cut;
            if (gone) { if (--L.n == 0) { CHECK(L.agg == 0);
                            rbook[o.sym][o.sd].erase(o.prc); }
                        rord.erase(ref); }
        };

        for (int it = 0; it < 300000; ++it) {
            const int op = static_cast<int>(rng() % 10);
            if (op < 4 || live.empty()) {                       // add 40%
                OrderRef ref = next_ref++;
                uint16_t sy = static_cast<uint16_t>(rng() % NSYM);
                uint8_t  sd = static_cast<uint8_t>(rng() % 2);
                // cluster prices per symbol so ladders stay in-window here
                PriceType prc = 1000000 + static_cast<PriceType>(sy) * 50000
                              + static_cast<PriceType>(rng() % 900) * 100;
                SizeType sz = 1 + static_cast<SizeType>(rng() % 250);
                CHECK(mds.on_add(ref, sy, static_cast<Side>(sd), prc, sz, ++t, ++u));
                rord[ref] = {sy, sd, prc, sz};
                rbook[sy][sd][prc].agg += sz;
                rbook[sy][sd][prc].n   += 1;
                live.push_back(ref);
            } else if (op < 7) {                                // partial reduce 30%
                OrderRef ref = live[rng() % live.size()];
                auto& o = rord[ref];
                if (o.sz > 1) {
                    SizeType cut = 1 + static_cast<SizeType>(rng() % (o.sz - 1));
                    CHECK(mds.on_reduce(ref, cut, ++t, ++u));
                    ref_reduce(ref, cut, false);
                }
            } else if (op < 9) {                                // full delete 20%
                std::size_t k = rng() % live.size();
                OrderRef ref = live[k];
                CHECK(mds.on_delete(ref, ++t, ++u));
                ref_reduce(ref, rord[ref].sz, true);
                live[k] = live.back(); live.pop_back();
            } else {                                            // replace 10%
                std::size_t k = rng() % live.size();
                OrderRef old_ref = live[k], new_ref = next_ref++;
                auto o = rord[old_ref];
                PriceType np = 1000000 + static_cast<PriceType>(o.sym) * 50000
                             + static_cast<PriceType>(rng() % 900) * 100;
                SizeType ns = 1 + static_cast<SizeType>(rng() % 250);
                CHECK(mds.on_replace(old_ref, new_ref, np, ns, ++t, ++u));
                ref_reduce(old_ref, o.sz, true);
                rord[new_ref] = {o.sym, o.sd, np, ns};
                rbook[o.sym][o.sd][np].agg += ns;
                rbook[o.sym][o.sd][np].n   += 1;
                live[k] = new_ref;
            }
            CHECK(mds.order_table().size() == rord.size());
            if (it % 977 == 0)
                for (uint16_t s = 0; s < NSYM; ++s)
                    verify_books(mds, s, rbook[s], 100);
        }
        for (uint16_t s = 0; s < NSYM; ++s) verify_books(mds, s, rbook[s], 100);
        std::printf("[churn] ok: orders=%zu max_chain=%zu pool_in_use=%zu "
                    "pages_in_use=%zu\n",
                    mds.order_table().size(), mds.order_table().max_chain(),
                    mds.order_table().pool_in_use(), mds.page_pool().in_use());
    }

    // =====================================================================
    // 2) black swan: tiny directory forces deep chains; delete most orders,
    //    overflow buckets must be reclaimed to the pool
    // =====================================================================
    {
        MarketDataSystem mds(1, /*peak_orders*/8 /*-> tiny dir*/, /*npages*/64,
                             /*swan*/4096);
        mds.configure_symbol(0, 100, 100, 4096);
        uint64_t t = 0, u = 0;
        const int N = 4000;
        for (int i = 1; i <= N; ++i)
            CHECK(mds.on_add(static_cast<OrderRef>(i), 0, Side::Buy,
                             500000 + (i % 97) * 100, 10, ++t, ++u));
        auto& ot = mds.order_table();
        std::printf("[swan] after %d adds: buckets=%zu max_chain=%zu "
                    "pool_in_use=%zu/%zu fails=%zu\n", N, ot.buckets(),
                    ot.max_chain(), ot.pool_in_use(), ot.pool_capacity(),
                    ot.insert_failures());
        CHECK(ot.max_chain() > 10);            // chains really formed
        CHECK(ot.pool_in_use() > 100);
        CHECK(ot.insert_failures() == 0);
        for (int i = 1; i <= N; ++i)           // every order still findable
            CHECK(ot.find(static_cast<OrderRef>(i)));
        for (int i = 1; i <= N - 20; ++i)      // drain most
            CHECK(mds.on_delete(static_cast<OrderRef>(i), ++t, ++u));
        std::printf("[swan] after drain: size=%zu pool_in_use=%zu max_chain=%zu\n",
                    ot.size(), ot.pool_in_use(), ot.max_chain());
        CHECK(ot.size() == 20);
        CHECK(ot.pool_in_use() <= 20);         // overflow buckets reclaimed
        for (int i = N - 19; i <= N; ++i)      // survivors intact
            CHECK(ot.find(static_cast<OrderRef>(i)));
    }

    // =====================================================================
    // 3) ladder overflow: far price beyond window -> overflow store; order
    //    at the overflow level lives a full lifecycle; best/roundlot merge
    // =====================================================================
    {
        MarketDataSystem mds(1, 1024, 64, 64);
        mds.configure_symbol(0, 100, 100, 4096);   // window = 4096 ticks
        uint64_t t = 0, u = 0;
        // anchor around 50.0000$ (500000); window covers +/- ~2048 ticks
        CHECK(mds.on_add(1, 0, Side::Buy, 500000, 60, ++t, ++u));
        CHECK(mds.on_add(2, 0, Side::Buy, 499900, 60, ++t, ++u));
        // stub bid at $0.01 — far below window -> overflow path
        CHECK(mds.on_add(3, 0, Side::Buy, 100, 500, ++t, ++u));
        CHECK(mds.csym(0).books[0].overflow_live() == 1);
        // best must still be the in-window 50.0000
        auto b = mds.best(0, Side::Buy);
        CHECK(b && *b == 500000);
        // roundlot(100): 60@500000 + 60@499900 -> price 499900
        auto r = mds.roundlot_best(0, Side::Buy);
        CHECK(r && r->price == 499900 && r->cum_size == 120 && r->levels_used == 2);
        // traversal reaches the overflow level last, with correct data
        std::vector<PriceType> ps;
        mds.for_each_level(0, Side::Buy, [&](PriceType p, const PrcLevelGeneric& L) {
            ps.push_back(p); if (p == 100) CHECK(L.agg_sz == 500);
            return true; });
        CHECK((ps == std::vector<PriceType>{500000, 499900, 100}));
        // partial reduce then delete the overflow order
        CHECK(mds.on_reduce(3, 200, ++t, ++u));
        CHECK(mds.level_at(0, Side::Buy, 100) && mds.level_at(0, Side::Buy, 100)->agg_sz == 300);
        CHECK(mds.on_delete(3, ++t, ++u));
        CHECK(mds.csym(0).books[0].overflow_live() == 0);
        CHECK(mds.level_at(0, Side::Buy, 100) == nullptr);
        // now make the overflow price the ONLY bid: best must come from ovf
        CHECK(mds.on_add(4, 0, Side::Sell, 90000000, 10, ++t, ++u)); // far ask -> ovf
        auto ab = mds.best(0, Side::Sell);
        CHECK(ab);   // ask side anchored by this add? anchor set on first add:
                     // this IS the first sell add -> in-window. Force overflow:
        CHECK(mds.on_add(5, 0, Side::Sell, 100, 10, ++t, ++u));      // far below anchor
        CHECK(mds.csym(0).books[1].overflow_live() == 1);
        ab = mds.best(0, Side::Sell);
        CHECK(ab && *ab == 100);               // overflow IS the best ask
        std::printf("[ovf] ok\n");
    }

    // =====================================================================
    // 4) page recycling: fill levels, drain them, pages return to pool
    // =====================================================================
    {
        MarketDataSystem mds(1, 4096, 32, 64);
        mds.configure_symbol(0, 100, 100, 8192);
        uint64_t t = 0, u = 0;
        OrderRef ref = 1;
        // spread levels across several pages (8192 ticks / 512 = 16 dir slots)
        for (int i = 0; i < 3000; ++i)
            CHECK(mds.on_add(ref++, 0, Side::Buy,
                             400000 + (i % 3000) * 100, 5, ++t, ++u));
        const std::size_t pg_peak = mds.page_pool().in_use();
        CHECK(pg_peak >= 5);
        for (OrderRef r2 = 1; r2 < ref; ++r2) CHECK(mds.on_delete(r2, ++t, ++u));
        std::printf("[pages] peak=%zu after-drain=%zu\n",
                    pg_peak, mds.page_pool().in_use());
        CHECK(mds.page_pool().in_use() == 0);  // every page recycled
        CHECK(!mds.best(0, Side::Buy));        // book empty, best recovered to none
    }

    // =====================================================================
    // 5) throughput: ITCH-like mix on one symbol
    // =====================================================================
    {
        MarketDataSystem mds(16, 2'000'000, 256, 4096);
        for (uint16_t s = 0; s < 16; ++s) mds.configure_symbol(s, 100, 100, 32768);
        std::mt19937_64 rng(1);
        uint64_t t = 0, u = 0;
        OrderRef next_ref = 1;
        std::vector<OrderRef> live; live.reserve(1u << 20);
        constexpr int OPS = 2'000'000;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < OPS; ++i) {
            const int op = static_cast<int>(rng() % 10);
            if (op < 4 || live.size() < 64) {
                OrderRef r2 = next_ref++;
                uint16_t sy = static_cast<uint16_t>(rng() % 16);
                mds.on_add(r2, sy, static_cast<Side>(rng() % 2),
                           1000000 + static_cast<PriceType>(rng() % 2000) * 100,
                           1 + static_cast<SizeType>(rng() % 200), ++t, ++u);
                live.push_back(r2);
            } else if (op < 8) {
                mds.on_reduce(live[rng() % live.size()], 1, ++t, ++u);
            } else {
                std::size_t k = rng() % live.size();
                mds.on_delete(live[k], ++t, ++u);
                live[k] = live.back(); live.pop_back();
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / OPS;
        std::printf("[bench] mixed add/reduce/delete: %.1f ns/msg "
                    "(%zu resident, max_chain=%zu)\n",
                    ns, mds.order_table().size(), mds.order_table().max_chain());
    }

    std::printf(failures ? "\n%d FAILURES\n" : "\nALL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
