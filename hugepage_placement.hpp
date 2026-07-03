// hugepage_placement.hpp — allocate a hugepage-backed (optionally NUMA-bound)
// region and placement-construct a StaticBucketed-style map into it, returning
// a RAII owner. Zero runtime allocation beyond the single mmap; the map is
// constructed WITHOUT touching its multi-GB storage (default-init of a trivial,
// valid-when-zeroed type), so the only page commits are the ones you ask for
// via `populate`.
//
// Requirements on Map:
//   * trivially default constructible (a zeroed object is a valid empty map);
//   * optionally exposes prefault() — used to fault the region on `populate`.
//
// Usage:
//   md::HugePageOptions opt; opt.page = md::HugeSize::MB2; opt.numa_node = 0;
//   auto book = md::make_hugepage_map<VenueMap>(opt);   // book-> is VenueMap*
//   book->upsert(ref, ...);
//   // region is munmap'd when `book` goes out of scope.
//
// Build: -std=c++23 -O3 -march=native. Linux only.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cerrno>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace md {

#ifndef MAP_HUGETLB
#  define MAP_HUGETLB 0x40000
#endif
#ifndef MAP_HUGE_SHIFT
#  define MAP_HUGE_SHIFT 26
#endif
#ifndef MPOL_BIND
#  define MPOL_BIND 2
#endif
#ifndef MPOL_MF_STRICT
#  define MPOL_MF_STRICT (1u << 0)
#endif
#ifndef MPOL_MF_MOVE
#  define MPOL_MF_MOVE (1u << 1)
#endif

enum class HugeSize : unsigned { MB2 = 21, GB1 = 30 };   // value == log2(page bytes)

struct HugePageOptions {
    HugeSize page           = HugeSize::MB2; // 2MB default; GB1 for very large maps
    int      numa_node      = -1;            // -1 => don't bind; else bind to node
    bool     populate       = true;          // fault the whole region now, not later
    bool     lock           = false;         // mlock (no swap; needs RLIMIT_MEMLOCK)
    bool     fallback_to_4k = false;         // if HUGETLB fails: normal pages + THP
};

// RAII owner. Move-only. Calls the map's destructor then munmaps on scope exit.
template <class Map>
class HugePageMap {
public:
    HugePageMap() = default;
    HugePageMap(Map* p, void* region, std::size_t bytes, bool huge, std::size_t page)
        : p_(p), region_(region), bytes_(bytes), page_(page), huge_(huge) {}
    HugePageMap(HugePageMap&& o) noexcept { move_from(o); }
    HugePageMap& operator=(HugePageMap&& o) noexcept {
        if (this != &o) { reset(); move_from(o); }
        return *this;
    }
    HugePageMap(const HugePageMap&)            = delete;
    HugePageMap& operator=(const HugePageMap&) = delete;
    ~HugePageMap() { reset(); }

    Map* operator->()       noexcept { return p_; }
    Map& operator*()        noexcept { return *p_; }
    Map* get()        const noexcept { return p_; }
    explicit operator bool()const noexcept { return p_ != nullptr; }

    [[nodiscard]] bool        huge_pages() const noexcept { return huge_; }
    [[nodiscard]] std::size_t bytes()      const noexcept { return bytes_; }
    [[nodiscard]] std::size_t page_bytes() const noexcept { return page_; }

private:
    void move_from(HugePageMap& o) noexcept {
        p_ = o.p_; region_ = o.region_; bytes_ = o.bytes_; page_ = o.page_; huge_ = o.huge_;
        o.p_ = nullptr; o.region_ = nullptr; o.bytes_ = 0;
    }
    void reset() noexcept {
        if (p_)      { p_->~Map(); p_ = nullptr; }
        if (region_) { ::munmap(region_, bytes_); region_ = nullptr; }
    }
    Map*        p_      = nullptr;
    void*       region_ = nullptr;
    std::size_t bytes_  = 0;
    std::size_t page_   = 0;
    bool        huge_   = false;
};

template <class Map>
HugePageMap<Map> make_hugepage_map(const HugePageOptions& opt = {}) {
    static_assert(std::is_trivially_default_constructible_v<Map>,
        "Map must be valid-when-zeroed: default construction must not touch storage");

    const std::size_t page_bytes = std::size_t{1} << static_cast<unsigned>(opt.page);
    const std::size_t bytes = (sizeof(Map) + page_bytes - 1) & ~(page_bytes - 1);

    auto fail = [&](const char* what) -> HugePageMap<Map> {
        throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
    };

    // Base hugetlb flags. When binding NUMA we must set the policy BEFORE the
    // pages fault, so we withhold MAP_POPULATE and fault via prefault() after
    // mbind. Without binding, MAP_POPULATE lets the kernel fault them up front.
    int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB
              | (static_cast<int>(opt.page) << MAP_HUGE_SHIFT);
    if (opt.populate && opt.numa_node < 0) flags |= MAP_POPULATE;

    void* region = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
    bool huge = true;

    if (region == MAP_FAILED) {
        if (!opt.fallback_to_4k)
            throw std::runtime_error(
                std::string("mmap MAP_HUGETLB failed (") + std::strerror(errno) +
                "). Reserve hugepages (e.g. sysctl vm.nr_hugepages) or set "
                "fallback_to_4k=true.");
        int f = MAP_PRIVATE | MAP_ANONYMOUS
              | (opt.populate && opt.numa_node < 0 ? MAP_POPULATE : 0);
        region = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, f, -1, 0);
        if (region == MAP_FAILED) return fail("mmap fallback failed");
        ::madvise(region, bytes, MADV_HUGEPAGE);   // transparent hugepages, best effort
        huge = false;
    }

    if (opt.numa_node >= 0) {
        unsigned long nodemask[16] = {0};
        nodemask[opt.numa_node / 64] |= (1ull << (opt.numa_node % 64));
        const long r = ::syscall(SYS_mbind, region, bytes, MPOL_BIND,
                                 nodemask, sizeof(nodemask) * 8,
                                 MPOL_MF_STRICT | MPOL_MF_MOVE);
        if (r != 0) { const int e = errno; ::munmap(region, bytes); errno = e;
                      if (!opt.fallback_to_4k) return fail("mbind to NUMA node failed"); }
    }

    if (opt.lock) ::mlock(region, bytes);   // best effort; ignore failure

    // Construct without writing to storage: DEFAULT-initialization (no parens/
    // braces). The trivial ctor is a no-op; the mapping is already zero-filled,
    // and a zeroed map is a valid empty map. (Value-init `Map()` would memset
    // the whole region — exactly what we avoid.)
    Map* p = ::new (region) Map;

    if (opt.populate) {
        if constexpr (requires (Map* m) { m->prefault(); }) p->prefault();
        else { volatile char* c = static_cast<volatile char*>(region);
               for (std::size_t o = 0; o < bytes; o += page_bytes) c[o] = c[o]; }
    }

    return HugePageMap<Map>(p, region, bytes, huge, page_bytes);
}

}  // namespace md
