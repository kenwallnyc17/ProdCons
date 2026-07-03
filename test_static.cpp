#include "static_bucketed_order_id_map.hpp"
#include <cstdio>
#include <random>
#include <unordered_map>
#include <vector>
using md::OrderRecord; using md::Put;

static int failures=0;
#define CHECK(c) do{ if(!(c)){std::printf("FAIL %s:%d %s\n",__FILE__,__LINE__,#c);++failures;} }while(0)

// Map lives in static storage, sized entirely at compile time.
// 4096 buckets, CAP=32 (AVX2), 200K-block pool -> insert-heavy forces overflow.
using BigMap = md::StaticBucketedOrderIdMap<OrderRecord, md::GroupAvx2,
                                            /*BucketBits*/12, /*PoolBlocks*/200000>;
constinit BigMap g_map{};                 // constant-initialised global
static_assert(BigMap::footprint_bytes() > 0);   // footprint known at compile time

// Tiny map to force pool exhaustion deterministically.
using TinyMap = md::StaticBucketedOrderIdMap<OrderRecord, md::GroupScalar,
                                             /*BucketBits*/7, /*PoolBlocks*/8>;
constinit TinyMap g_tiny{};

int main(){
    std::printf("compile-time footprint: %.3f GB  (%zu buckets x %zuB + %zu pool blocks)\n",
        double(BigMap::footprint_bytes())/1e9, BigMap::num_buckets(),
        BigMap::block_bytes(), BigMap::pool_blocks());

    // ---- insert-heavy: drive the static map into its overflow pool ----
    const std::size_t N = 300000;          // > 4096*32 = 131072 inline slots
    std::vector<std::uint64_t> keys; keys.reserve(N);
    std::mt19937_64 rng(31337);
    { std::unordered_map<std::uint64_t,char> s;
      while(keys.size()<N){ std::uint64_t k=rng(); if(!k||s.count(k))continue; s[k]=1; keys.push_back(k);} }
    for(std::size_t i=0;i<N;++i){
        Put p=g_map.upsert(keys[i],{0,(std::uint32_t)i,0,0,0});
        CHECK(p==Put::Inserted);
    }
    CHECK(g_map.size()==N);
    CHECK(g_map.max_chain()>1);            // chains formed
    CHECK(g_map.pool_in_use()>0);          // pool actually used
    CHECK(g_map.insert_failures()==0);
    for(std::size_t i=0;i<N;++i){ auto* r=g_map.find(keys[i]); CHECK(r && r->qty==(std::uint32_t)i); }
    std::printf("global map after %zu inserts: max_chain=%zu pool_in_use=%zu fails=%zu\n",
        N, g_map.max_chain(), g_map.pool_in_use(), g_map.insert_failures());

    // ---- delete half: exercise swap-remove in pool blocks + reclaim ---
    std::size_t pool_before=g_map.pool_in_use();
    for(std::size_t i=0;i<N;i+=2){ OrderRecord out{}; CHECK(g_map.erase(keys[i],&out)); CHECK(out.qty==(std::uint32_t)i); }
    CHECK(g_map.size()==N-(N+1)/2);
    for(std::size_t i=0;i<N;++i){ auto* r=g_map.find(keys[i]);
        if(i%2==0) CHECK(r==nullptr); else CHECK(r && r->qty==(std::uint32_t)i); }
    std::printf("after deleting half: size=%zu pool_in_use=%zu (was %zu, reclaimed %zu)\n",
        g_map.size(), g_map.pool_in_use(), pool_before, pool_before-g_map.pool_in_use());
    CHECK(g_map.pool_in_use() <= pool_before);

    // ---- pool exhaustion is REPORTED, not allocated -------------------
    std::unordered_map<std::uint64_t,std::uint32_t> tref;
    std::size_t inserted=0, exhausted=0;
    for(std::uint64_t k=1; k<=200000; ++k){
        Put p=g_tiny.upsert(k,{0,(std::uint32_t)k,0,0,0});
        if(p==Put::Inserted){ ++inserted; tref[k]=(std::uint32_t)k; }
        else if(p==Put::PoolExhausted){ ++exhausted; }
    }
    std::printf("tiny map: inserted=%zu pool_exhausted=%zu pool_in_use=%zu/%zu\n",
        inserted, exhausted, g_tiny.pool_in_use(), TinyMap::pool_blocks());
    CHECK(exhausted>0);
    CHECK(g_tiny.pool_in_use()==TinyMap::pool_blocks());
    for(auto [k,q]:tref){ auto* r=g_tiny.find(k); CHECK(r && r->qty==q); }  // accepted ones intact

    std::printf(failures?"\n%d FAILURES\n":"\nALL TESTS PASSED\n", failures);
    return failures?1:0;
}
