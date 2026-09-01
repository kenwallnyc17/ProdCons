#include "efvi_md.hpp"
#include <chrono>
#include <cstdio>
#include <random>
using namespace md::net;
static double now_ns(){ return std::chrono::duration<double,std::nano>(
    std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(){
    for (unsigned n : {4u, 8u, 16u, 32u, 64u}) {
        ChannelTable t; for (unsigned i = 0; i < n; ++i) t.add(0xE9000100 + i, (uint16_t)(30000 + i), (uint8_t)i);
        std::mt19937_64 rng(1); std::vector<uint64_t> keys;
        for (int i = 0; i < 4096; ++i) { unsigned c = (unsigned)(rng() % n); keys.push_back((uint64_t(0xE9000100 + c) << 16) | (30000 + c)); }
        volatile int sink = 0; const int R = 4'000'000;
        double t0 = now_ns();
        for (int i = 0; i < R; ++i) sink += t.lookup(keys[i & 4095]);
        std::printf("channels=%2u  lookup %.2f ns\n", n, (now_ns() - t0) / R);
    }
}
