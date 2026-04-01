#ifndef MTONVARIABLE_2025_H_INCLUDED
#define MTONVARIABLE_2025_H_INCLUDED

///to do:
// cached w and r idx
// N readers
//returns a pointer to the data rather than a copy

#include "Definitions.h"


/// KMW  ??? are the below still true (12/2/25)
///   * incomplete, has problem with setting widx_ to new value before data written
///   * either zero out data chunk after reads (Aeron does this) or serialize the setting of widx_ and use a reserve_widx_
namespace MKTDATASYSTEM::CONTAINERS
{

struct MtoNVariable_2025 final
{
    MtoNVariable_2025(const size_t num_bytes);
    ~MtoNVariable_2025();

    bool write(const char* d, const size_t len) noexcept;
    void read1(char* d, size_t& len) noexcept;
    void readN(char* d, size_t& len) noexcept;

private:
    const size_t cap_;
    const size_t mask_;
    constexpr static size_t sizeof_len{sizeof(size_t)};
    char* buf_{};

    alignas(2*cacheline_size_bytes) atomic<size_t> widx_{};
    alignas(2*cacheline_size_bytes) atomic<size_t> ridx_{};

    size_t get_widx(const size_t len) noexcept;
};

size_t MtoNVariable_2025::get_widx(const size_t bytes) noexcept
{
    const size_t need{roundup_bytes.template operator()<cacheline_size_bytes>(sizeof_len + bytes)};

    size_t expected_widx{}, new_widx{};
    bool try_this{};

    /// false sharing
    /// write at end or start at beginning
    /// slow reader
    do
    {
        expected_widx = widx_.load(memory_order_acquire);
        const size_t curr_ridx = ridx_.load(memory_order_acquire); /// use cached

     //   const size_t relative_widx{expected_widx & mask_};

       /* size_t end_offset{};

        /// can it fit at end? if not, write at beginning
        if(relative_widx + need > cap_)
        {
            end_offset = cap_ - relative_widx;
        }
        */

        try_this = false;

        ///slow read check
        if(expected_widx + need /*+ end_offset*/ - curr_ridx <= cap_ )
        {
            new_widx = expected_widx + need;
          //  new_widx = roundup_bytes<cacheline_size>(new_widx);/// round up to next cacheline
            try_this = true;
        }

    }while(try_this && widx_.compare_exchange_weak(expected_widx, new_widx) == false);

    return expected_widx;
}

MtoNVariable_2025::MtoNVariable_2025(const size_t num_bytes) :
    cap_(roundup_bytes_pow2.template operator()<cacheline_size_bytes>(num_bytes)), mask_(cap_-1)
{
    buf_ = new (std::align_val_t(page_size_bytes)) char[cap_]{};
}

MtoNVariable_2025::~MtoNVariable_2025()
{
  //  delete[] buf_;
    ::operator delete[](buf_, cap_, std::align_val_t(page_size_bytes));
}

bool MtoNVariable_2025::write(const char* d, const size_t len) noexcept
{
    if(len > cap_) return false;

    const size_t widx{get_widx(len)};
    const size_t relative_widx{widx & mask_};

    if(relative_widx + len < cap_)[[likely]]///single write
    {
        memcpy(&buf_[relative_widx+sizeof_len], d, len);
    }
    else
    {
        auto frst{cap_ - (relative_widx +sizeof_len)};
        memcpy(&buf_[relative_widx+sizeof_len], d, frst);
        memcpy(&buf_[0], &d[frst+sizeof_len], len-frst);
    }

    /// the above must be completed before writing the len
    atomic_thread_fence(memory_order_release);

    memcpy(&buf_[relative_widx], static_cast<const void*>(&len), sizeof_len);/// always at least a cacheline available

#if 1
    cout << "\n in Queue write: " << __FILE__ << ":" << __LINE__ << " : len, relative_widx, cap, buf_ = " << len << ", " << relative_widx << ", " << cap_ << endl;
    for(int i = relative_widx; i < relative_widx+len; ++i)
        cout << buf_[i];

    cout << endl;
#endif
    return true;
}

/// ??? zero out data on buf_ after read. it will prevent erroneous data being read in the future
void MtoNVariable_2025::read1(char* d, size_t& len) noexcept
{
    auto curr_ridx{ridx_.load(memory_order_acquire)};
    auto curr_widx{widx_.load(memory_order_acquire)};///use cached value

    while(curr_ridx == curr_widx)
    {
        curr_widx = widx_.load(memory_order_acquire);
    }
//cout << "\n in Queue read1: " << __FILE__ << ":" << __LINE__ << " : curr_ridx, curr_widx = " << curr_ridx << ", " << curr_widx << endl;
    const size_t relative_ridx{curr_ridx & mask_};

    memcpy(static_cast<void*>(&len), &buf_[relative_ridx], sizeof_len);

//cout << "\n in Queue read1: " << __FILE__ << ":" << __LINE__ << " : len, relative_ridx, cap_ = " << len << ", " << relative_ridx << ", " << cap_ << endl;

    /// possible write that overlaps to beginning of buf_
    if(len <= (cap_ - (relative_ridx+sizeof_len)))[[likely]]
    {
        memcpy(d, &buf_[relative_ridx+sizeof_len], len);
    }
    else
    {
        auto frst{cap_ - (relative_ridx +sizeof_len)};
        memcpy(d, &buf_[relative_ridx+sizeof_len], frst);
        memcpy(d+frst, &buf_[0], len - frst);
    }

    ridx_.store(roundup_bytes.template operator()<cacheline_size_bytes>(curr_ridx + sizeof_len + len), memory_order_release);
}

void test_m2nvariable_1()
{
    cout << "\n MTONVARIABLE::test_m2nvariable_1()" << endl;

    MtoNVariable_2025 m2n1(129);

    char buf[10]{};
    size_t L{};

    m2n1.write("k", 1);
    m2n1.write("e", 1);
    m2n1.read1(&buf[0], L);
    m2n1.read1(&buf[1], L);
    m2n1.write("n", 1);
    m2n1.read1(&buf[2], L);

    cout << "buf = " << buf << endl;

    cout << "\n end MTONVARIABLE::test_m2nvariable_1()" << endl;
}

}

#endif // MTONVARIABLE_2025_H_INCLUDED
