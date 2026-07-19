#include "mdsys.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
using namespace md;

static int failures=0;
#define CHECK(c) do{ if(!(c)){std::printf("FAIL %s:%d %s\n",__FILE__,__LINE__,#c);++failures;} }while(0)

using T = MarketDataSystem::TickMsg::T;
static MarketDataSystem::TickMsg mk(T t, OrderRef r, OrderRef r2, uint16_t sy,
    Side sd, PriceType p, SizeType sz, uint64_t& tt, uint64_t& uu){
    return {t, r, r2, sy, sd, p, sz, ++tt, ++uu};
}

static void compare_systems(const MarketDataSystem& a, const MarketDataSystem& b,
                            uint16_t nsym){
    CHECK(a.order_table().size() == b.order_table().size());
    for(uint16_t s=0; s<nsym; ++s) for(unsigned sd=0; sd<2; ++sd){
        std::vector<std::tuple<PriceType,uint64_t,uint32_t>> la, lb;
        a.for_each_level(s,(Side)sd,[&](PriceType p,const PrcLevelGeneric&L){
            la.push_back({p,L.agg_sz,L.num_orders}); return true;});
        b.for_each_level(s,(Side)sd,[&](PriceType p,const PrcLevelGeneric&L){
            lb.push_back({p,L.agg_sz,L.num_orders}); return true;});
        CHECK(la==lb);
        CHECK(a.best(s,(Side)sd)==b.best(s,(Side)sd));
        auto ra=a.roundlot_best(s,(Side)sd), rb=b.roundlot_best(s,(Side)sd);
        CHECK(ra.has_value()==rb.has_value());
        if(ra&&rb){ CHECK(ra->price==rb->price && ra->cum_size==rb->cum_size); }
    }
}

int main(){
    // ---- equivalence: batched pipeline == per-message, random batch sizes --
    {
        const uint16_t NSYM=4;
        MarketDataSystem A(NSYM,30000,256,2048), B(NSYM,30000,256,2048);
        for(uint16_t s=0;s<NSYM;++s){ A.configure_symbol(s,100,100,8192);
                                      B.configure_symbol(s,100,100,8192); }
        std::mt19937_64 rng(99);
        uint64_t t=0,u=0; OrderRef next=1;
        std::vector<OrderRef> live; std::vector<SizeType> lsz;
        std::vector<MarketDataSystem::TickMsg> batch;
        for(int round=0; round<12000; ++round){
            batch.clear();
            const std::size_t bs = 1 + rng()%32;           // packet of 1..32 msgs
            for(std::size_t k=0;k<bs;++k){
                int op=(int)(rng()%10);
                if(op<4||live.empty()){
                    OrderRef r=next++;
                    uint16_t sy=(uint16_t)(rng()%NSYM);
                    PriceType p=1000000+(PriceType)sy*40000+(PriceType)(rng()%700)*100;
                    SizeType z=2+(SizeType)(rng()%200);
                    batch.push_back(mk(T::Add,r,0,sy,(Side)(rng()%2),p,z,t,u));
                    live.push_back(r); lsz.push_back(z);
                } else if(op<7){
                    std::size_t k2=rng()%live.size();
                    if(lsz[k2]>1){ SizeType c=1+(SizeType)(rng()%(lsz[k2]-1));
                        batch.push_back(mk(T::Reduce,live[k2],0,0,Side::Buy,0,c,t,u));
                        lsz[k2]-=c; }
                } else if(op<9){
                    std::size_t k2=rng()%live.size();
                    batch.push_back(mk(T::Delete,live[k2],0,0,Side::Buy,0,0,t,u));
                    live[k2]=live.back(); live.pop_back();
                    lsz[k2]=lsz.back(); lsz.pop_back();
                } else {
                    std::size_t k2=rng()%live.size();
                    OrderRef nr=next++;
                    PriceType p=1000000+(PriceType)(rng()%NSYM)*40000
                               +(PriceType)(rng()%700)*100;
                    SizeType z=2+(SizeType)(rng()%200);
                    batch.push_back(mk(T::Replace,live[k2],nr,0,Side::Buy,p,z,t,u));
                    live[k2]=nr; lsz[k2]=z;
                }
            }
            // A: per-message handlers; B: batched pipeline. Same stream.
            for(auto& x:batch){
                switch(x.type){
                case T::Add:     A.on_add(x.ref,x.symst,x.side,x.prc,x.sz,x.ft,x.uu); break;
                case T::Reduce:  A.on_reduce(x.ref,x.sz,x.ft,x.uu); break;
                case T::Delete:  A.on_delete(x.ref,x.ft,x.uu); break;
                case T::Replace: A.on_replace(x.ref,x.ref2,x.prc,x.sz,x.ft,x.uu); break; }
            }
            B.process_batch(batch.data(), batch.size());
            if(round%700==0) compare_systems(A,B,NSYM);
        }
        compare_systems(A,B,NSYM);
        std::printf("[equiv] ok: %zu resident, max_chain=%zu\n",
            B.order_table().size(), B.order_table().max_chain());
    }

    // ---- intra-batch hazards: same-ref lifecycle inside ONE packet --------
    {
        MarketDataSystem S(1,256,32,32);
        S.configure_symbol(0,100,100,4096);
        uint64_t t=0,u=0;
        std::vector<MarketDataSystem::TickMsg> b;
        // add r=1; partial reduce; reduce-to-zero (deletes); reduce again (must
        // fail cleanly); add r=2; delete r=2; replace nonexistent (fails clean)
        b.push_back(mk(T::Add,    1,0,0,Side::Buy, 500000, 100, t,u));
        b.push_back(mk(T::Reduce, 1,0,0,Side::Buy, 0,       30, t,u));
        b.push_back(mk(T::Reduce, 1,0,0,Side::Buy, 0,       70, t,u)); // -> delete
        b.push_back(mk(T::Reduce, 1,0,0,Side::Buy, 0,        5, t,u)); // stale: no-op
        b.push_back(mk(T::Add,    2,0,0,Side::Buy, 500100,  40, t,u));
        b.push_back(mk(T::Delete, 2,0,0,Side::Buy, 0,        0, t,u));
        b.push_back(mk(T::Replace,3,4,0,Side::Buy, 500200,  10, t,u)); // absent old
        S.process_batch(b.data(), b.size());
        CHECK(S.order_table().size()==0);
        CHECK(!S.best(0,Side::Buy));
        CHECK(S.page_pool().in_use()==0);
        std::printf("[hazard] ok\n");
    }

    // ---- bench: per-message vs batched pipeline at large resident --------
    {
        auto build=[&](MarketDataSystem& S, std::vector<MarketDataSystem::TickMsg>& ops){
            std::mt19937_64 rng(7); uint64_t t=0,u=0; OrderRef next=1;
            std::vector<OrderRef> live; live.reserve(1u<<21);
            // warm-up resident book
            for(int i=0;i<1'500'000;++i){
                OrderRef r=next++;
                uint16_t sy=(uint16_t)(rng()%16);
                S.on_add(r,sy,(Side)(rng()%2),
                    1000000+(PriceType)(rng()%3000)*100,
                    50+(SizeType)(rng()%100),++t,++u);
                live.push_back(r);
            }
            ops.clear(); ops.reserve(2'000'000);
            for(int i=0;i<2'000'000;++i){
                int op=(int)(rng()%10);
                if(op<4){ OrderRef r=next++;
                    uint16_t sy=(uint16_t)(rng()%16);
                    ops.push_back(mk(T::Add,r,0,sy,(Side)(rng()%2),
                        1000000+(PriceType)(rng()%3000)*100,
                        50+(SizeType)(rng()%100),t,u));
                    live.push_back(r);
                } else if(op<8){
                    ops.push_back(mk(T::Reduce,live[rng()%live.size()],0,0,
                        Side::Buy,0,1,t,u));
                } else {
                    std::size_t k=rng()%live.size();
                    ops.push_back(mk(T::Delete,live[k],0,0,Side::Buy,0,0,t,u));
                    live[k]=live.back(); live.pop_back();
                }
            }
        };
        std::vector<MarketDataSystem::TickMsg> ops;
        double per_ns, batch_ns;
        {
            MarketDataSystem S(16,4'000'000,512,8192);
            for(uint16_t s=0;s<16;++s) S.configure_symbol(s,100,100,32768);
            build(S,ops);
            auto t0=std::chrono::steady_clock::now();
            for(auto& x:ops){ switch(x.type){
                case T::Add: S.on_add(x.ref,x.symst,x.side,x.prc,x.sz,x.ft,x.uu); break;
                case T::Reduce: S.on_reduce(x.ref,x.sz,x.ft,x.uu); break;
                case T::Delete: S.on_delete(x.ref,x.ft,x.uu); break;
                default: break; } }
            auto t1=std::chrono::steady_clock::now();
            per_ns=std::chrono::duration<double,std::nano>(t1-t0).count()/ops.size();
        }
        {
            MarketDataSystem S(16,4'000'000,512,8192);
            for(uint16_t s=0;s<16;++s) S.configure_symbol(s,100,100,32768);
            build(S,ops);
            constexpr std::size_t B=32;                    // ~MoldUDP64 packet
            auto t0=std::chrono::steady_clock::now();
            for(std::size_t i=0;i<ops.size();i+=B)
                S.process_batch(&ops[i], std::min(B, ops.size()-i));
            auto t1=std::chrono::steady_clock::now();
            batch_ns=std::chrono::duration<double,std::nano>(t1-t0).count()/ops.size();
        }
        std::printf("[bench] per-message %.1f ns/msg   batched+prefetch %.1f ns/msg  (%.0f%%)\n",
            per_ns, batch_ns, 100.0*(per_ns-batch_ns)/per_ns);
    }

    std::printf(failures?"\n%d FAILURES\n":"\nALL TESTS PASSED\n",failures);
    return failures?1:0;
}
