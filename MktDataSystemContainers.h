#ifndef MKTDATASYSTEMCONTAINERS_H_INCLUDED
#define MKTDATASYSTEMCONTAINERS_H_INCLUDED

#include "Definitions.h"
#include "MktDataSystemBaseDefinitions.h"
#include "MemoryCustomization.h"

#include "MtoNVariable_2025.h"

namespace MKTDATASYSTEM::CONTAINERS
{
using namespace MKTDATASYSTEM::DEFINITIONS;

/**
    specialization for a map Value that is a pointer
*/

template<typename V, typename K>
concept ValidHashMapPtrValues = (is_pointer_v<V>
                                 && requires(V v){{v->id()}->same_as<K>;});

/// POWOF2_buckets use this for defining the number of buckets
/// INITVALCNT_bucket = the initial size (how many deafult VAL's) for each bucket -- can grow via rehash
/// to do:
//      * limits of POWOF2_buckets
//      * add memory pool for alloc's
//
template<unsigned_integral KEY, ValidHashMapPtrValues<KEY> VAL>
struct FastUIntHashMap
{
private:
    using Bucket = vector<VAL>;
    using VAL_t = remove_pointer_t<decay_t<remove_pointer_t<VAL>>>;

    Bucket* pbuckets{};

    const size_t bucket_cnt{};
    const KEY mask{};

    CUSTOMMEMORY::MemoryPool<alignof(VAL_t)> mempool_;

    /// if k exists, replace with v and return old v -- for user to delete if they want to
    /// NOTE: this shouln't be used for adding a new VAL to the map. It skips the mempool usage.
    VAL insert(KEY k, VAL v) noexcept
    {
        KEY bkt_idx{k & mask};

        for(auto& el : pbuckets[bkt_idx])
        {
            /// if key exists, replace the pointer and return value
            if(el->id() == k)
            {
                VAL oldV{el};
                el = v;
                return oldV;
            }
        }

        /// to see if this fails, could check if before size == after size, in which case it means failure --> return nullptr
        pbuckets[bkt_idx].push_back(v);/// may rehash the bucket, also, may throw

        return v;
    }

public:
    FastUIntHashMap(const uint16_t POWOF2_buckets, const size_t INITVALCNT_bucket,
                    const size_t MemPoolBlocksInit, const size_t MemPoolBlocksGrowthFactor) noexcept :
        bucket_cnt(1<<POWOF2_buckets), mask(bucket_cnt - 1),
        mempool_(sizeof(remove_pointer_t<decay_t<remove_pointer_t<VAL>>>), MemPoolBlocksInit,
                 MemPoolBlocksGrowthFactor) /// << is lower precedence than -
    {
        /// use 'special way' to delete this
        pbuckets = new (std::align_val_t(page_size_bytes)) Bucket[bucket_cnt];

        for(size_t i = 0; i < bucket_cnt; ++i)
        {
            pbuckets[i].reserve(INITVALCNT_bucket);
         //   cout << el.capacity() << " : ";
        }
       // cout << endl;
    }

    FastUIntHashMap(const FastHashMapInputs& ob) noexcept:
        FastUIntHashMap(ob.POWOF2_buckets, ob.INITVALCNT_bucket,
                        ob.MemPoolBlocks.Init, ob.MemPoolBlocks.GrowthFactor) {}

    ~FastUIntHashMap()
    {
        /// special way to delete
        ::operator delete[](pbuckets, bucket_cnt, std::align_val_t(page_size_bytes));
    }

    VAL find(KEY k) noexcept
    {
        KEY bkt_idx{k & mask};

        for(auto& el : pbuckets[bkt_idx])
        {
            if(el->id() == k)
            {
                return el;
            }
        }

        return nullptr;
    }

    bool erase(KEY k) noexcept
    {
        KEY bkt_idx{k & mask};

        bool found{};
        for(auto it = pbuckets[bkt_idx].begin(); it != pbuckets[bkt_idx].end(); ++it)
        {
            /// if key exists, replace the pointer and return value
            if((*it)->id() == k)
            {
                mempool_.deallocate(reinterpret_cast<void*>(*it));

                /// maybe write my own shift down ??
                pbuckets[bkt_idx].erase(it);

                found == true;
                break;
            }
        }

        return found;
    }

    /// if k doesn't exist, allocate new value object, insert it
    /// ??? what do I really want to do here (try_* usually means if it already exists, then don't overwrite)
    template</*typename F,*/ typename... VArgs>
    VAL try_emplace(KEY k, /*F&& f_onfail,*/ VArgs&&... vrgs) noexcept
    {
        KEY bkt_idx{k & mask};

        for(auto& el : pbuckets[bkt_idx])
        {
            /// if key exists, ??? delete old pointer
            if(el->id() == k)
            {
              //  f_onfail(k, el, forward<VArgs>(vrgs)...);
             //   delete el;
              //  el = new VAL_t(forward<VArgs>(vrgs)...);
                return el;
            }
        }
cout << __FILE__ << " : "<< __LINE__  << " : " << type_name<VAL_t>() << " : " << sizeof...(VArgs)<<endl;

        VAL nv{new (mempool_.allocate()) VAL_t(forward<VArgs>(vrgs)...)};

        pbuckets[bkt_idx].push_back(nv);/// may rehash the bucket

        return nv;
    }
};

///
/// specialized hash map that uses IP and port as a key and the value can be anything
///
template<typename IPType = uint32_t>
struct SockAddr
{
    IPType ip;
    uint16_t port;
};

template<typename IPType>
bool operator<(const SockAddr<IPType>& lh, const SockAddr<IPType>& rh)
{
    return lh.port < rh.port;
}

template<typename IPType>
bool operator==(const SockAddr<IPType>& lh, const SockAddr<IPType>& rh)
{
    return lh.port == rh.port && lh.ip == rh.ip;
}

/// maybe add an allocator for the vector
///     * it would be shared among all instances of the vector
///     * could be stack memeory, because it will be small
/// ??? make num_bits a compile time var
template<typename IPType, typename V>
struct SockAddrHashMap
{
    struct KeyValPair
    {
        SockAddr<IPType>    sa;
        V           val;
    };

    static constexpr uint8_t num_bits{8};
    vector<KeyValPair> arr_[1<<num_bits];
    uint16_t mask_{(1<<num_bits) -1};

    static auto idx = [](const uint16_t port){return port & mask_;};

    const V* find(const SockAddr<IPType>& sa) const
    {
        for(auto& el : arr_[idx(sa.port)])
        {
            if(el.sa == sa)
                return &el.val;
        }

        return nullptr;
    }

    void insert(const KeyValPair& kv)
    {
        arr_[idx(kv.sa)].insert(kv);
    }
};


template<typename IPType, typename V, size_t N>
struct SockAddrFlatMap
{
    //using KeyArray = std::array<SockAddr<IPType>, N>;
    //using ValArray = std::array<V, N>;
    using KeyArray = std::vector<SockAddr<IPType>>;
    using ValArray = std::vector<V>;

    /// if N < 10 (??), just do a linear search through the KeyArray
    const V* find(const SockAddr<IPType>& sa) const
    {
        if(auto search = fmap_.find(sa); search != fmap_.end())
            return &search->second;

        return nullptr;
    }

    void insert(const SockAddr<IPType>& k, const V& v)//const pair<SockAddr<IPType>, V>& kv)
    {
        fmap_.emplace(make_pair(k,v));//move(kv));
        //fmap_[k] = v;
    }
private:
    std::flat_map<SockAddr<IPType>, V, less<SockAddr<IPType>>, KeyArray, ValArray> fmap_;
};

}

#endif // MKTDATASYSTEMCONTAINERS_H_INCLUDED
