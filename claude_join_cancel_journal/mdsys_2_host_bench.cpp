// mdsys_2_host_bench.cpp — what a feed handler pays per message when it calls
// md::CancelTactic and md::JoinTactic (mdsys_2.hpp [4], [5]).
//
// A host in one translation unit of its own: ONE handler, ONE call site per
// tactic entry point (on_place for our own Adds included) and nothing else
// of the tactics in sight. That is the shape in which the compiler is free
// to inline whatever the header lets it into the handler, so it is the shape
// in which the header's inlining decisions show: the frame the handler gets,
// and what every message pays for it. mdsys_2_cancel_bench part 14 has the
// same rows inside the large bench, where other call sites change what the
// inliner does.
//
// Rows, per message kind (Add / Delete / Execute; removals carry kQGone), 8
// symbols round-robin, state in cache: a symbol with no orders of ours; a
// pinned one; one order alone in its level; one order behind a round lot, so
// (b) is armed, with the messages at another price; the same with them at
// our price. Cells: median / p99 TSC ticks, timer overhead subtracted.
//
// As it stands it uses no API newer than the first delivery of the two
// tactics, so the same file builds against any header since for a
// before/after on your own compiler and flags: the header is included with
// <>, so -I picks the version (a "" include would always take the one next
// to this file). With -DHOST_QEV the handler passes on_book_update the queue's
// event word (QEv: handle, sizes, flags) instead of the bare flags — what a
// host whose orders can be recognised late passes, so that the journal holds
// what on_place_late needs; the tactic does the same work for either.
//
// Build: g++ -std=c++23 -O3 -march=graniterapids -DNDEBUG -I<dir of mdsys_2.hpp> mdsys_2_host_bench.cpp -o host_bench
// Run:   taskset -c <isolated cpu> ./host_bench <label>
// The handler's frame:
//   objdump -d -C host_bench | awk '/^[0-9a-f]+ <handle\(/,/^$/' | grep -cE 'ymm|and +\$0xffffffffffffffe0,%rsp|%fs:0x28'
// counts its 256-bit instructions, its stack realignment and its stack-protector references: 0 with this header.
#include <mdsys_2.hpp>
#include <x86intrin.h>
#include <algorithm>
#include <cstdio>
#include <vector>
using namespace md;
struct alignas(64) NbboRec { Nbbo q; char pad[48]; };
static std::vector<NbboRec> g_rec(64);
Nbbo md::getNBBO(uint16_t s) noexcept { return g_rec[s].q; }
Nbbo md::getVenueRoundLotBBO(uint16_t s) noexcept { return g_rec[s].q; }
static std::vector<SymbolState> g_st(64);
SymbolState& md::getSymbolState(uint16_t s) noexcept { return g_st[s]; }
RlQuote md::getVenueRoundLot(uint16_t, Side) noexcept { return {0, 0}; }
struct Msg { uint16_t sym; BookEvent ev; Side sd; PriceType px; FeedTimeType ts; uint8_t qf; uint32_t sz; PriceType prev;
             uint64_t ours; uint32_t qh;
#ifdef HOST_QEV
             QEv qe{};                                 // what the book's queue call returned for this message
#endif
};
static uint64_t g_fired = 0;
[[gnu::noinline]] void handle(CancelTactic& ct, JoinTactic& jt, const Msg& m) {
    const auto cb = [](CancelTactic::Handle, uint64_t, CancelTactic::Decision) { ++g_fired; };
    if (m.ev == BookEvent::Add) {
        if (m.ours) ct.on_place(m.ours, m.sym, m.sd, m.px, m.ts, m.qh, cb);
        else jt.on_add(m.sym, m.sd, m.px, m.sz, m.prev, m.ts, [](uint16_t, Side, const JoinSide&) { ++g_fired; });
    } else if (m.ev == BookEvent::Execute) {
        jt.on_execute(m.sym, m.sd, m.px, m.sz);
    }
#ifdef HOST_QEV
    ct.on_book_update(m.sym, m.ev, m.sd, m.px, m.ts, m.qe, cb);
#else
    ct.on_book_update(m.sym, m.ev, m.sd, m.px, m.ts, m.qf, cb);
#endif
}
static inline uint64_t tb() { _mm_lfence(); const uint64_t t = __rdtsc(); _mm_lfence(); return t; }
static inline uint64_t te() { unsigned a; const uint64_t t = __rdtscp(&a); _mm_lfence(); return t; }
int main(int argc, char** argv) {
    const char* label = argc > 1 ? argv[1] : "?";
    { const uint64_t end = __rdtsc() + 1'500'000'000ull; volatile uint64_t x = 1; while (__rdtsc() < end) x = x * 3 + 1; }
    std::vector<uint32_t> ov(400000);
    for (auto& x : ov) { const uint64_t a = tb(); const uint64_t b = te(); x = uint32_t(b - a); }
    std::sort(ov.begin(), ov.end());
    const uint64_t oh = ov[ov.size() / 2];
    constexpr unsigned NS = 32;                        // 8 symbols per row, round-robin (hot)
    const PriceType B = 1'000'000, A = 1'000'100;
    QBlockPool pool(64);
    std::vector<QBlockList> lists(NS);
    CancelParams p;
    p.aggr_prints_n = 2'000'000'000;                   // (d) counts, never fires
    CancelTactic ct(NS, 64, pool, p);
    JoinTactic jt(NS);
    FeedTimeType t = 1'000'000'000;
    for (unsigned s = 0; s < NS; ++s) {
        const uint16_t y = uint16_t(s);
        ct.set_symbol(y, 100, 100); g_rec[y].q = {B, A}; g_st[y].round_lot = 100;
        if (s >= 8 && s < 16) ct.set_tracking(y, true);
        if (s >= 16) {                                 // our order, registered through the handler
            if (s >= 24) lists[s].push(pool, (1ull << 40) + s, 200);      // behind a round lot: (b) armed
            const uint32_t qh = lists[s].push(pool, 1000 + s, 100);
            Msg m{y, BookEvent::Add, Side::Buy, B - 300, t, 0, 100, B, 1000 + s, qh};
#ifdef HOST_QEV
            m.qe = QEv{qh, 0, 100, kQNew};
#endif
            handle(ct, jt, m);
        }
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
                Msg m{uint16_t(std::min(row, 3u) * 8 + (i & 7)), kinds[k], Side::Buy, px, t += 1'000,
                      uint8_t(k ? kQGone : 0), 100, B, 0, 0};
#ifdef HOST_QEV
                m.qe = k ? QEv{0x20u + (i & 7), 100, 0, kQGone} : QEv{0x20u + (i & 7), 0, 100, kQNew};
#endif
                asm volatile("" : : "r"(&m) : "memory");
                const uint64_t a = tb();
                handle(ct, jt, m);
                const uint64_t b = te();
                if (i >= 200'000) v.push_back(uint32_t(b - a));
            }
            std::sort(v.begin(), v.end());
            auto at = [&](double f) { double x = double(v[size_t(f * double(v.size() - 1))]) - double(oh); return x < 0 ? 0 : x; };
            std::printf(" | %s %3.0f/%3.0f", names[k], at(0.5), at(0.99));
        }
        std::printf("%s\n", g_fired ? "  ** FIRED **" : "");
    }
    return 0;
}
