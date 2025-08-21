#ifndef USEFUL_H_INCLUDED
#define USEFUL_H_INCLUDED


inline constexpr uint64_t cacheline_size_bytes{64};
inline constexpr uint64_t page_size_bytes{1024};//4096};
inline constexpr uint64_t huge_page_size_bytes{2*1024*1024};

constexpr auto powof2 = [](const size_t n, const size_t min = 1)
{
    size_t p{min};

    while(p < n)
        p *= 2;

    return p;
};

constexpr auto roundupto = []<uint64_t roundto>(const size_t n)
{
    const size_t r{(n/roundto) * roundto};
    return r + ((r == n) ? 0 : roundto);
};


template<convertible_to<uint64_t> T>
struct alignas(16) atomic_16b
{
    uint64_t hi{};
    uint64_t lo{};

   // atomic_16b(T hi, T lo) : hi(hi), lo(lo){}

    [[gnu::always_inline]]
    bool cas(atomic_16b& exp, atomic_16b& rep)
    {
         bool result;
        __asm__ __volatile__
        (
            "lock cmpxchg16b %1\n\t"
            "setz %0"       // on gcc6 and later, use a flag output constraint instead
            : "=q" ( result )
            , "+m" ( *this )
            , "+d" ( exp.lo )
            , "+a" ( exp.hi )
            : "c" ( rep.lo )
            , "b" ( rep.hi )
            : "cc", "memory" // compile-time memory barrier.  Omit if you want memory_order_relaxed compile-time ordering.
        );
        return result;
    }
};


#endif // USEFUL_H_INCLUDED
