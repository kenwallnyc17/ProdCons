#ifndef PRODCONSTESTING_H_INCLUDED
#define PRODCONSTESTING_H_INCLUDED

#include "ProdConsGeneric.h"

namespace PRODCONSGENERIC
{
/// //////////////////////////////////////////////////////////////////////////////////////////////////////
/// //////////////////////////////////////////////////////////////////////////////////////////////////////
struct D1
{
    int prod;
    size_t v;
    friend ostream& operator<<(ostream& os, const D1& d)
    {
        os << d.prod << ":" << d.v << " ";
        return os;
    }
};

struct D1_nomoves
{
    int prod;
    int v;
    friend ostream& operator<<(ostream& os, const D1_nomoves& d)
    {
        os << d.prod << ":" << d.v << " ";
        return os;
    }

    D1_nomoves(D1_nomoves&&) = delete;
    D1_nomoves& operator=(D1_nomoves&&) = delete;
};


/// use this for testing Fixed size data containers
template<template<typename,size_t,auto>typename PRODCONS, typename D, size_t N, int PRODS = 2, int CONS = 2>
struct ProdConsTester
{
    ProdConsTester() = default;

    void producer(int core, int pidx, deque<D> ds)
    {
        ::SetThreadAffinityMask(::GetCurrentThread(), 1 << core);

        cout << "core, pidx = " << core << ", " << pidx << endl;

        size_t i{0};
        while(exit == false && i < ds.size())
        {
            ds[i].prod = pidx;

            if(pc.prod(ds[i]) == false)
                continue;

            ++i;

           // std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    void consumer(int core, const int cons)
    {
      //  int core = 5;
        ::SetThreadAffinityMask(::GetCurrentThread(), 1 << core);

        int i{};
        while(exit == false )
        {
            D d;
            if(pc.cons(d) == false)
                continue;

            ///serialize writes
            {
                static mutex mtx;

                unique_lock lk(mtx);

                cout << "cons, d = " << cons << ", " << d << endl;
            }
            ++i;

       //     int in;
        //    cin >> in;

            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }

        cout << "num read = " << i << endl;
    }

    void operator()(deque<D> ds)
    {
        cout << "ProdConsTester::operator()()" << endl;

        int core = 8;
        ::SetThreadAffinityMask(::GetCurrentThread(), 1 << core);

        deque<std::thread> joinable;

        joinable.emplace_back(&producer, this, 2, 0, ds);

        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        joinable.emplace_back(&consumer, this, 11, 2);

        if constexpr (PRODS == 2)
            joinable.emplace_back(&producer, this, 5, 1, ds);

        if constexpr (CONS == 2)
            joinable.emplace_back(&consumer, this, 9, 3);

        int in;
        cin >> in;

        exit = true;

        for(auto& el : joinable)
        {
            el.join();
        }

        cout << " end ProdConsTester::operator()()" << endl;
    }

    inline static volatile bool exit{};
    inline static auto toexit = [](){return exit;};

    PRODCONS<D,N,toexit> pc;
};

/// variable sized data objects
struct VDBase
{
    const int type;
    size_t len{};
    VDBase(const int t) : type(t) {}
};

template<int t, int N>
struct VD : VDBase
{
    VD() : VDBase(t) {len = sizeof(VD<t,N>);}

    char d[N];
};

using VD1 = VD<1,76>;
using VD2 = VD<2,145>;
using VD3 = VD<3,911>;

/// use this for testing Variable size data containers
template<template<typename,size_t>typename PRODCONS, size_t N, derived_from<VDBase> ...Args>
struct ProdConsVariableTester
{
    void producer(int core, int pidx, const int repeats)
    {
        ::SetThreadAffinityMask(::GetCurrentThread(), 1 << core);

        cout << "core, pidx = " << core << ", " << pidx << endl;

        int i{0};
        while(exit == false && i < repeats)
        {
            /// write each element of the tuple to the data buffer, repeat this as provided
            apply([this](auto&&... as)
                {
                    auto loop = [this](auto&& v)
                    {
                        /// keep trying to write the update until achieved
                        while(exit == false && pc.prod(reinterpret_cast<const char*>(&v), v.len) == false)
                            ;
                    };
                    (loop(as), ...);

                }, tpl);

            ++i;
        }
    }

    void consumer(int core, int cidx)
    {
        ::SetThreadAffinityMask(::GetCurrentThread(), 1 << core);

        char data[4*1024];
        int i{};
        while(exit == false )
        {
            uint32_t len;

            pc.cons(&data[0], len);
            if(len == 0)
                continue;

            cout << "consumer: read len, type = " << cidx << ": " << len << ", "
                << reinterpret_cast<VDBase*>(&data[0])->type << endl;

            ++i;

            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        cout << "read cnt = " << i << endl;
    }

    void operator()(const int repeats, const int num_prods = 1, const int num_cons = 1)
    {
        cout << "ProdConsTester::operator()()" << endl;

        int core = 8;
        ::SetThreadAffinityMask(::GetCurrentThread(), 1 << core);

        deque<std::thread> joinable;

        joinable.emplace_back(&producer, this, 2, 0, repeats);

        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        joinable.emplace_back(&consumer, this, 11, 2);

        if (num_prods == 2)
            joinable.emplace_back(&producer, this, 5, 1, repeats);

        if (num_cons == 2)
            joinable.emplace_back(&consumer, this, 9, 3);

        int in;
        cin >> in;

        exit = true;

        for(auto& el : joinable)
        {
            el.join();
        }

        cout << " end ProdConsTester::operator()()" << endl;//<< num_consumed << endl;
    }

    PRODCONS<char,N> pc;
    volatile bool exit{};

    tuple<Args...> tpl;
};

}

#endif // PRODCONSTESTING_H_INCLUDED
