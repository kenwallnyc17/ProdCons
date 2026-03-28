#ifndef DEFINITIONS_H_INCLUDED
#define DEFINITIONS_H_INCLUDED

inline constexpr uint64_t cacheline_size_bytes{64};
inline constexpr uint64_t page_size_bytes{1024};//4096};
inline constexpr uint64_t huge_page_size_bytes{2*1024*1024};

constexpr auto powof2 = [](const size_t n, const size_t min = 1) noexcept
{
    size_t p{min};

    while(p < n)
        p *= 2;

    return p;
};

constexpr auto roundupto = []<uint64_t roundto = alignof(std::max_align_t)>(const size_t n) noexcept
{
    const size_t r{(n/roundto) * roundto};
    return r + ((r == n) ? 0 : roundto);
};

template<size_t val>
concept is_cacheline_alignable = requires{ requires (val%cacheline_size_bytes) == 0;};


///lowest acceptable denominaator is cacheline size
/// assumes denomsize is pow2
constexpr auto roundup_bytes = []<size_t denomsize>(const size_t ask) noexcept requires is_cacheline_alignable<denomsize>
{
    return ((ask + (denomsize-1))/denomsize) * denomsize;
};

constexpr auto roundup_bytes_pow2 = []<size_t denomsize>(const size_t ask) noexcept requires is_cacheline_alignable<denomsize>
{
    size_t up{(ask + (denomsize-1))/denomsize};

    size_t p2 = 1;
    while(p2 < up)
        p2 *= 2;

    return p2 * denomsize;
};

constexpr size_t MTULEN{1500};

#endif // DEFINITIONS_H_INCLUDED
