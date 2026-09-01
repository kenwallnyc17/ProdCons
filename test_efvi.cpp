// test_efvi.cpp — mock-backed test + bench for efvi_md.hpp
#include "efvi_md.hpp"
#include <chrono>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <vector>
using namespace md::net;

static int failures = 0;
#define CHECK(c) do{ if(!(c)){ std::printf("FAIL %s:%d %s\n",__FILE__,__LINE__,#c); ++failures; } }while(0)
static double now_ns(){ return std::chrono::duration<double,std::nano>(
    std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- frame builder: Ethernet II / IPv4 / UDP / MoldUDP64 ---------------
struct Msg { uint64_t seq; uint8_t chan; uint8_t side; uint16_t len; };
static std::vector<uint8_t> build_frame(uint32_t dst_ip, uint16_t dst_port, uint64_t seq,
                                        const std::vector<uint16_t>& mlens, uint8_t chan,
                                        uint16_t count_override = 0xFFFE) {
    std::vector<uint8_t> f(42, 0);
    f[12]=0x08; f[13]=0x00; f[14]=0x45; f[22]=64; f[23]=17;
    uint32_t ipbe = htonl(dst_ip); std::memcpy(&f[30], &ipbe, 4);
    uint32_t src = htonl(0x0A000001); std::memcpy(&f[26], &src, 4);
    uint16_t pbe = htons(dst_port); std::memcpy(&f[36], &pbe, 2);
    // mold header
    const char sess[10] = {'S','E','S','S','I','O','N','0','0','1'};
    f.insert(f.end(), sess, sess+10);
    uint64_t sbe = std::byteswap(seq); const uint8_t* sp=(const uint8_t*)&sbe; f.insert(f.end(), sp, sp+8);
    uint16_t cnt = count_override == 0xFFFE ? (uint16_t)mlens.size() : count_override;
    uint16_t cbe = std::byteswap(cnt); const uint8_t* cp=(const uint8_t*)&cbe; f.insert(f.end(), cp, cp+2);
    uint64_t s = seq;
    for (uint16_t ml : mlens) {
        uint16_t lbe = std::byteswap(ml); const uint8_t* lp=(const uint8_t*)&lbe; f.insert(f.end(), lp, lp+2);
        // payload: [chan][seq lo..] then filler — lets the sink verify attribution
        std::vector<uint8_t> body(ml, 0xAB);
        body[0] = chan; std::memcpy(&body[1], &s, 8 < ml-1 ? 8 : ml-1);
        f.insert(f.end(), body.begin(), body.end());
        ++s;
    }
    uint16_t udp_len = (uint16_t)(f.size() - 34);
    uint16_t ulbe = htons(udp_len); std::memcpy(&f[38], &ulbe, 2);
    uint16_t tot = (uint16_t)(f.size() - 14); uint16_t tbe = htons(tot); std::memcpy(&f[16], &tbe, 2);
    if (f.size() < 60) f.resize(60, 0);
    return f;
}

struct Sink {
    std::vector<Msg> got;
    std::map<uint8_t, uint64_t> last_seq;
    uint64_t bad_attr = 0, ts_sum = 0;
    void on_msg(uint8_t chan, uint8_t side, uint64_t seq, const uint8_t* p, uint32_t len, uint64_t ts) noexcept {
        got.push_back({seq, chan, side, (uint16_t)len});
        uint64_t s = 0; std::memcpy(&s, p+1, 8);
        if (p[0] != chan || s != seq) ++bad_attr;      // cross-channel leak / wrong seq
        ts_sum += ts;
        auto it = last_seq.find(chan);
        if (it != last_seq.end() && seq <= it->second) ++bad_attr;   // must be strictly increasing
        last_seq[chan] = seq;
    }
};

struct Chan { uint32_t ip[2]; uint16_t port[2]; };
static const Chan kChans[4] = {
    {{0xE9000101, 0xE9000201}, {30001, 30002}},
    {{0xE9000102, 0xE9000202}, {30003, 30004}},
    {{0xE9000103, 0xE9000203}, {30005, 30006}},
    {{0xE9000104, 0xE9000204}, {30007, 30008}},
};

template <class R>
static void setup(R& rx, const Config& cfg) {
    CHECK(rx.add_port(1, 0, cfg) == 0);   // side A
    CHECK(rx.add_port(2, 1, cfg) == 1);   // side B
    for (uint8_t c = 0; c < 4; ++c) {
        CHECK(rx.add_channel(c, 0, kChans[c].ip[0], kChans[c].port[0]) == 0);
        CHECK(rx.add_channel(c, 1, kChans[c].ip[1], kChans[c].port[1]) == 0);
    }
}

int main() {
    // ===== directed: dups, side race, gap, heartbeat, isolation ============
    for (bool merge : {false, true}) {
        Config cfg; cfg.event_merge = merge; cfg.rx_ring = 64; cfg.refill_batch = 8;
        Sink sink; Receiver<Sink> rx(sink); setup(rx, cfg);
        auto inj = [&](uint8_t chan, uint8_t side, uint64_t seq, std::vector<uint16_t> ml, uint16_t cnt = 0xFFFE) {
            auto f = build_frame(kChans[chan].ip[side], kChans[chan].port[side], seq, ml, chan, cnt);
            CHECK(mock_efvi_inject(rx.port(side).vi(), f.data(), (uint16_t)f.size(), 1'000'000 + seq) == 0);
        };
        auto sync = [&]{ mock_efvi_flush_multi(rx.port(0).vi()); mock_efvi_flush_multi(rx.port(1).vi());
                         while (rx.poll_once() > 0) {} };
        inj(0,0,1,{20,30}); sync();                  // A: 1,2 delivered
        inj(0,1,1,{20,30}); sync();                  // B: dup
        inj(0,1,3,{25});    sync();                  // B first: 3 from side B
        inj(0,0,3,{25});    sync();                  // A: dup
        inj(0,0,10,{40});   sync();                  // gap 4..9 (6), 10 delivered
        inj(0,0,11,{}, 0);  sync();                  // heartbeat at next: no gap
        inj(0,1,13,{}, 0);  sync();                  // heartbeat: gap 11..12 (2)
        inj(1,0,100,{10}); inj(2,1,7,{10}); inj(3,0,1,{10}); sync();   // other channels
        inj(0,0,4,{10});    sync();                  // late lower seq: dup
        std::vector<uint64_t> c0; for (auto& m : sink.got) if (m.chan == 0) c0.push_back(m.seq);
        CHECK((c0 == std::vector<uint64_t>{1,2,3,10}));
        CHECK(rx.stats().dups == 3);
        CHECK(rx.stats().gaps == 2 && rx.stats().gap_msgs == 6 + 2);
        CHECK(rx.stats().heartbeats == 2);
        CHECK(rx.arbiter(0).next_seq() == 13);
        CHECK(sink.got.size() == 4 + 3 && sink.bad_attr == 0);
        CHECK(sink.got[0].side == 0 && sink.got[2].side == 1);   // side attribution
        CHECK(rx.port(0).stats().unknown_dst == 0 && rx.port(0).stats().bad_hdr == 0);
        // unknown group + malformed frame are counted, never delivered
        auto junk = build_frame(0xE9FFFFFF, 1, 1, {10}, 9);
        mock_efvi_inject(rx.port(0).vi(), junk.data(), (uint16_t)junk.size(), 0);
        std::vector<uint8_t> bad(80, 0); bad[12]=0x86; bad[13]=0xDD;   // IPv6 ethertype
        mock_efvi_inject(rx.port(0).vi(), bad.data(), 80, 0);
        mock_efvi_inject_discard(rx.port(0).vi());
        sync();
        CHECK(rx.port(0).stats().unknown_dst == 1 && rx.port(0).stats().bad_hdr == 1
              && rx.port(0).stats().discards == 1);
        CHECK(sink.got.size() == 7);
        std::printf("[directed%s] ok\n", merge ? " merge" : "");
    }

    // ===== randomized: drops/reorder per side, reference arbitration ========
    {
        Config cfg; cfg.event_merge = true; cfg.rx_ring = 256; cfg.refill_batch = 32;
        Sink sink; Receiver<Sink> rx(sink); setup(rx, cfg);
        std::mt19937_64 rng(2026);
        struct Ev { uint8_t chan, side; uint64_t seq; std::vector<uint16_t> ml; uint64_t key; };
        std::vector<Ev> arrivals;
        uint64_t next_seq[4] = {1,1,1,1};
        for (int i = 0; i < 20000; ++i) {
            uint8_t c = (uint8_t)(rng()%4);
            std::vector<uint16_t> ml; int nm = 1 + (int)(rng()%3);
            for (int j=0;j<nm;++j) ml.push_back((uint16_t)(12 + rng()%40));
            uint64_t s = next_seq[c]; next_seq[c] += (uint64_t)nm;
            for (uint8_t side = 0; side < 2; ++side) {
                if (rng()%10 == 0) continue;                       // 10% loss per side
                uint64_t jitter = rng()%5;                         // reorder window
                arrivals.push_back({c, side, s, ml, (uint64_t)i*8 + jitter*2 + side});
            }
        }
        std::stable_sort(arrivals.begin(), arrivals.end(), [](auto&a, auto&b){ return a.key < b.key; });
        // reference: same rule, applied in the order the receiver actually
        // sees packets — per 40-packet round, port A drained then port B
        // (a single poll thread over multiple VIs IS sequential across ports)
        uint64_t ref_next[4] = {0,0,0,0}; bool synced[4] = {false,false,false,false};
        std::vector<std::pair<uint8_t,uint64_t>> ref;
        auto apply = [&](const Ev& e){
            if (!synced[e.chan]) { ref_next[e.chan] = e.seq; synced[e.chan] = true; }
            for (size_t j = 0; j < e.ml.size(); ++j) {
                uint64_t s = e.seq + j;
                if (s >= ref_next[e.chan]) { ref.push_back({e.chan, s}); ref_next[e.chan] = s + 1; }
            } };
        for (size_t b = 0; b < arrivals.size(); b += 40) {
            size_t e = std::min(arrivals.size(), b + 40);
            for (uint8_t side = 0; side < 2; ++side)
                for (size_t i = b; i < e; ++i) if (arrivals[i].side == side) apply(arrivals[i]);
        }
        size_t idx = 0;
        while (idx < arrivals.size()) {
            for (int k = 0; k < 40 && idx < arrivals.size(); ++k, ++idx) {
                auto& e = arrivals[idx];
                auto f = build_frame(kChans[e.chan].ip[e.side], kChans[e.chan].port[e.side], e.seq, e.ml, e.chan);
                CHECK(mock_efvi_inject(rx.port(e.side).vi(), f.data(), (uint16_t)f.size(), e.key) == 0);
            }
            mock_efvi_flush_multi(rx.port(0).vi()); mock_efvi_flush_multi(rx.port(1).vi());
            while (rx.poll_once() > 0) {}
        }
        CHECK(sink.got.size() == ref.size());
        bool same = sink.got.size() == ref.size();
        for (size_t i = 0; same && i < ref.size(); ++i)
            same = sink.got[i].chan == ref[i].first && sink.got[i].seq == ref[i].second;
        CHECK(same);
        CHECK(sink.bad_attr == 0);
        CHECK(rx.port(0).stats().no_desc == 0 && rx.port(1).stats().no_desc == 0);
        CHECK(rx.port(0).vi()->no_desc_drops == 0);
        std::printf("[random] ok: %zu msgs delivered, dups=%llu gaps=%llu\n", sink.got.size(),
            (unsigned long long)rx.stats().dups, (unsigned long long)rx.stats().gaps);
    }

    // ===== descriptor starvation is counted when not polled ================
    {
        Config cfg; cfg.rx_ring = 16; cfg.refill_batch = 4; cfg.event_merge = false;
        Sink sink; Receiver<Sink> rx(sink); setup(rx, cfg);
        auto f = build_frame(kChans[0].ip[0], kChans[0].port[0], 1, {10}, 0);
        int ok = 0; for (int i = 0; i < 40; ++i) ok += mock_efvi_inject(rx.port(0).vi(), f.data(), (uint16_t)f.size(), 0) == 0;
        CHECK(ok == 16 && rx.port(0).vi()->no_desc_drops == 24);
        while (rx.poll_once() > 0) {}
        ok = 0; for (int i = 0; i < 16; ++i) ok += mock_efvi_inject(rx.port(0).vi(), f.data(), (uint16_t)f.size(), 0) == 0;
        CHECK(ok == 16);                                         // ring fully refilled
        std::printf("[starve] ok\n");
    }

    // ===== bench: poll+classify+demux+arbitrate+deliver ====================
    {
        Config cfg; cfg.event_merge = true; cfg.rx_ring = 1024; cfg.refill_batch = 32;
        struct NullSink { uint64_t n = 0, sum = 0;
            void on_msg(uint8_t, uint8_t, uint64_t seq, const uint8_t*, uint32_t, uint64_t) noexcept { ++n; sum += seq; } };
        NullSink ns; Receiver<NullSink> rx(ns); setup(rx, cfg);
        std::vector<std::vector<uint8_t>> frames[4];
        for (uint8_t c = 0; c < 4; ++c) for (uint64_t s = 1; s <= 512; s += 2)
            frames[c].push_back(build_frame(kChans[c].ip[0], kChans[c].port[0], s, {24, 30}, c));
        const int rounds = 4000, per = 512;   // 2.05M packets
        double inject_ns = 0, total_ns = 0;
        uint64_t seqbase = 0;
        for (int r = 0; r < rounds; ++r) {
            double t0 = now_ns();
            for (int i = 0; i < per; ++i) {
                auto& f = frames[i & 3][(i >> 2) & 255];
                // patch seq to keep arbiter advancing (each frame carries 2 msgs)
                uint64_t s = std::byteswap(seqbase + (uint64_t)(i >> 2) * 2 + 1);
                std::memcpy(&f[52], &s, 8);
                CHECK(mock_efvi_inject(rx.port(0).vi(), f.data(), (uint16_t)f.size(), 0) == 0);
            }
            mock_efvi_flush_multi(rx.port(0).vi());
            double t1 = now_ns();
            while (rx.poll_once() > 0) {}
            double t2 = now_ns();
            inject_ns += t1 - t0; total_ns += t2 - t1;
            seqbase += 512;
        }
        const double pk = (double)rx.port(0).stats().pkts;
        CHECK(rx.port(0).stats().pkts == (uint64_t)rounds * per);
        CHECK(ns.n == (uint64_t)rounds * per * 2);
        std::printf("[bench] receive path: %.1f ns/pkt (%.1f ns/msg) over %.0f pkts; mock inject %.1f ns/pkt\n",
            total_ns / pk, total_ns / (pk * 2), pk, inject_ns / pk);
    }
    std::printf(failures ? "\n%d FAILURES\n" : "\nALL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
