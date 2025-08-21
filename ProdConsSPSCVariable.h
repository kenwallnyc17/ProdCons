#ifndef PRODCONSSPSCVARIABLE_H_INCLUDED
#define PRODCONSSPSCVARIABLE_H_INCLUDED

#include "ProdConsGeneric.h"

namespace PRODCONSSPECIFIC
{
using namespace PRODCONSGENERIC;

/// expect that we have a single writer and a single reader
/// the size of the data write is variable
template<same_as<char> T, size_t N>
class ProdConsSPSCVariable final : public ProdConsVariableData<T,N>
{
public:
    ProdConsSPSCVariable() = default;

    ProdConsSPSCVariable(const ProdConsSPSCVariable&) = delete;
    ProdConsSPSCVariable& operator=(const ProdConsSPSCVariable&) = delete;
    ProdConsSPSCVariable(ProdConsSPSCVariable&&) = delete;
    ProdConsSPSCVariable& operator=(ProdConsSPSCVariable&&) = delete;

    bool prod(const char* p, uint32_t len) noexcept override;
    void cons(char* p, uint32_t& len) noexcept override;

    /// inject this, otherwise it's blocked by the prod(..) of this class
    using ProdConsVariableData<T,N>::prod;

private:

    using ProdConsAlignedDataBuffer<T,N>::write;
    using ProdConsAlignedDataBuffer<T,N>::read;
    using ProdConsAlignedDataBuffer<T,N>::sz_;
    using ProdConsAlignedDataBuffer<T,N>::buf_mask_;

    size_t widx_cached_{}, ridx_cached_{};
    alignas(cacheline_size_bytes) atomic<size_t> widx_{}, ridx_{};

    constexpr static size_t szof_len_{sizeof(uint32_t)};
};

template<same_as<char> T, size_t N>
bool ProdConsSPSCVariable<T,N>::prod(const char* p, uint32_t len) noexcept
{
  //  cout << "ProdConsSPSCVariable<T,N>::prod(const char* p, uint16_t len) : " /*<< p << ", "*/ << len << endl;

    if(p == nullptr || len == 0) return false;

    size_t to_wrt_len{roundupto.template operator()<cacheline_size_bytes>(szof_len_ + len)};

    /// scenarios:
    /// (i) room to write at end of buffer
    /// (ii) room to write with overlap to begin of buffer
    /// (iii) no room to write, wait/return ??

    const size_t curr_widx{widx_.load(memory_order_relaxed)};/// I am only one who writes widx

    /// see if the reader is too slow to do a write, if so, return
    if(curr_widx + to_wrt_len > ridx_cached_ + sz_)// check global indices
    {
        ridx_cached_ = ridx_.load(memory_order_acquire);

        ///(iii)
        if(curr_widx + to_wrt_len > ridx_cached_ + sz_)
            return false;
    }

    const size_t curr_buf_widx{curr_widx&buf_mask_};

//    cout << "(i), to_wrt_len, (curr_buf_widx + szof_len_) = " << to_wrt_len << ", " << (curr_buf_widx + szof_len_) << endl;
    /// (i)
    if(curr_buf_widx + to_wrt_len <= sz_)[[likely]]
    {
        write(curr_buf_widx, reinterpret_cast<const char*>(&len), szof_len_);
        write(curr_buf_widx + szof_len_, p, len);

        widx_.store(curr_widx + to_wrt_len, memory_order_release);
    }
    /// (ii) need to write at beginning of buffer
    else
    {
   //     cout << "(ii)" << endl;
        size_t bytes_consumed{};
        /// can I at least write the len at the end?? -- if any room at all there will be at least 1 cacheline
        if(curr_buf_widx + szof_len_ <= sz_)
        {
 //           cout << "if((curr_buf_widx + szof_len_) < sz_)   " << (curr_buf_widx + szof_len_) << endl;
            /// ok to write the len even if we don't end up writing the data because in that event, I won't update widx_
           write(curr_buf_widx, reinterpret_cast<const char*>(&len), szof_len_);

           bytes_consumed = sz_ - curr_buf_widx; /// wasn't enough room to write data, but could be more than 1 cacheline

           /// update this
           to_wrt_len = roundupto.template operator()<cacheline_size_bytes>(len);
        }

        /// is there room at the begin of buffer to write the data
        if(to_wrt_len > (ridx_cached_&buf_mask_))
        {
            /// update cached ridx
            ridx_cached_ = ridx_.load(memory_order_acquire);

            if(to_wrt_len > (ridx_cached_&buf_mask_))
                return false;
        }

 //       cout << "bytes_consumed, len, (curr_widx + to_wrt_len + bytes_consumed) = "
//            << bytes_consumed << ", " << len << ", " << (curr_widx + to_wrt_len + bytes_consumed) << endl;

        /// did I write the len already ??
        if(bytes_consumed == 0)
        {
            write(0, reinterpret_cast<const char*>(&len), szof_len_);
            write(szof_len_, p, len);
        }
        else
            write(0, p, len);

        widx_.store(curr_widx + to_wrt_len + bytes_consumed, memory_order_release);
    }

    return true;
}

template<same_as<char> T, size_t N>
void ProdConsSPSCVariable<T,N>::cons(char* p, uint32_t& len) noexcept
{
    size_t curr_ridx{ridx_.load(memory_order_relaxed)};

    /// see if the buffer is empty
    if(curr_ridx == widx_cached_)
    {
        widx_cached_ = widx_.load(memory_order_acquire);

 //       cout << "widx_cached_ = " << widx_cached_ << endl;

        if(curr_ridx == widx_cached_)
        {
            len = 0;
            return;
        }
    }

    const size_t curr_buf_ridx{curr_ridx&buf_mask_};

    /// have something to read
    read(curr_buf_ridx, reinterpret_cast<char*>(&len), szof_len_);

//    cout << "len, widx_cached_, curr_ridx, curr_buf_ridx, (((curr_buf_ridx+szof_len_) + len) < sz_) = "
//        << len << ", " << widx_cached_ << ", " << curr_ridx << ", " << curr_buf_ridx << ", "
//        << (((curr_buf_ridx+szof_len_) + len) < sz_) << endl;

    /// can I read the data from before the end of the buffer?
    if(curr_buf_ridx+szof_len_ + len <= sz_)
    {
//        cout << "if(((curr_buf_ridx+szof_len_) + len) < sz_)   : " << (curr_buf_ridx+szof_len_) << endl;
        read(curr_buf_ridx+szof_len_, p, len);

        ridx_.store(curr_ridx + roundupto.template operator()<cacheline_size_bytes>(szof_len_+len), memory_order_release);
    }
    else/// len written at end of buffer and the data at the beginning
    {
 //       cout << "NOT if((((curr_ridx&buf_mask_)+szof_len_) + len) < sz_)" << endl;
        /// since ridx_ will be at the beginning of a cacheline, can just skip the usage of szof_len_ in the next 2 lines
        const size_t skipped_data{sz_- ((curr_ridx+szof_len_)&buf_mask_)};

        curr_ridx += szof_len_ + skipped_data;

        read(curr_ridx&buf_mask_, p, len);

        ridx_.store(curr_ridx + roundupto.template operator()<cacheline_size_bytes>(len), memory_order_release);
    }
}

void test_spscvariable_1()
{
    cout << "\nPRODCONSSPECIFIC::test_spscvariable_1 \n" << endl;

    if(false)
    {
        ProdConsSPSCVariable<char, 146> pc1;

        cout << "pc1.cap() = " << pc1.cap() << endl;

        for(uint64_t i = 1; i < 4; ++i)
        {
            pc1.prod(i);
        }
       // uint64_t d1{1};//numeric_limits<uint64_t>::max()};


        uint32_t len;
        char d[256]{};
        char* pd = &d[0];

        while(true)
        {
            pc1.cons(pd, len);
            if(pd != nullptr && len != 0)
            {
                uint64_t* pu{reinterpret_cast<uint64_t*>(pd)};
                cout << "len, *reinterpret_cast<uint64_t*>(pd) = " << len << ", " << *pu << endl;
            }
            else
            {
                break;
            }
        }

        for(uint64_t i = 4; i < 7; ++i)
        {
            pc1.prod(i);
        }

        while(true)
        {
            pc1.cons(pd, len);
            if(pd != nullptr && len != 0)
            {
                uint64_t* pu{reinterpret_cast<uint64_t*>(pd)};
                cout << "len, *reinterpret_cast<uint64_t*>(pd) = " << len << ", " << *pu << endl;
            }
            else
            {
                break;
            }
        }

        for(uint64_t i = 7; i < 10; ++i)
        {
            pc1.prod(i);
        }

        while(true)
        {
            pc1.cons(pd, len);
            if(pd != nullptr && len != 0)
            {
                uint64_t* pu{reinterpret_cast<uint64_t*>(pd)};
                cout << "len, *reinterpret_cast<uint64_t*>(pd) = " << len << ", " << *pu << endl;
            }
            else
            {
                break;
            }
        }

    }

    if(false)
    {
        ProdConsSPSCVariable<char, 146> pc1;

        cout << "pc1.cap() = " << pc1.cap() << endl;

        uint32_t len;
        char d[256]{};
        char* pd = &d[0];

     //   uint64_t iarr1[3]{1};
      //  uint64_t iarr2[] = {1,9,8};
     //   uint64_t* iarr{new uint64_t[5]{}};
        uint64_t i{1};

       // pc1.prod(17llu);/// type of U, u = long long unsigned int, long long unsigned int&&
      //  pc1.prod(move(i));/// type of U, u = long long unsigned int, long long unsigned int&&

       // pc1.prod(iarr1);/// type of U, u = long long unsigned int (&)[3], long long unsigned int (&)[3]
       // pc1.prod(iarr2);/// type of U, u = long long unsigned int (&)[3], long long unsigned int (&)[3]
       // pc1.prod(iarr);/// type of U, u = long long unsigned int*&, long long unsigned int*&
        pc1.prod(i);
        pc1.cons(pd, len);
        if(len != 0)
        {
            uint64_t* pu{reinterpret_cast<uint64_t*>(pd)};
            cout << "len, *reinterpret_cast<uint64_t*>(pd) = " << len << ", " << *pu << endl;
        }

        pd = &d[0];
        pc1.prod(i=2);
        pc1.cons(pd, len);
        if(len != 0)
        {
            uint64_t* pu{reinterpret_cast<uint64_t*>(pd)};
            cout << "len, *reinterpret_cast<uint64_t*>(pd) = " << len << ", " << *pu << endl;
        }

        pd = &d[0];
        pc1.prod(i=3);
        pc1.cons(pd, len);
        if(len != 0)
        {
            uint64_t* pu{reinterpret_cast<uint64_t*>(pd)};
            cout << "len, *reinterpret_cast<uint64_t*>(pd) = " << len << ", " << *pu << endl;
        }

        pd = &d[0];
        pc1.prod(i=4);
        pc1.cons(pd, len);
        if(len != 0)
        {
            uint64_t* pu{reinterpret_cast<uint64_t*>(pd)};
            cout << "len, *reinterpret_cast<uint64_t*>(pd) = " << len << ", " << *pu << endl;
        }

        pd = &d[0];
        pc1.prod(i=5);
        pc1.cons(pd, len);
        if(len != 0)
        {
            uint64_t* pu{reinterpret_cast<uint64_t*>(pd)};
            cout << "len, *reinterpret_cast<uint64_t*>(pd) = " << len << ", " << *pu << endl;
        }
    }


    cout << "\n end PRODCONSGENERIC::test_spscvariable_1 \n" << endl;
}

void test_spscvariable_2()
{
    cout << "\nPRODCONSSPECIFIC::test_spscvariable_2 \n" << endl;

    alignas(cacheline_size_bytes) atomic_16b<uint64_t> nxtwr_lastrd_(87, 34);
    __m128i got_data = _mm_load_si128((__m128i const*)&nxtwr_lastrd_);

    atomic_16b<uint64_t>* pat{reinterpret_cast<atomic_16b<uint64_t>*>(&got_data)};
    cout << pat->hi << " : " << pat->lo << endl;

    try
    {
        ProdConsVariableTester<ProdConsSPSCVariable, 4011, VD1, VD2, VD3> pcv;

        pcv(5);
    }
    catch(...)
    {

    }

    cout << "\n end PRODCONSSPECIFIC::test_spscvariable_2 \n" << endl;
}

}

#endif // PRODCONSSPSCVARIABLE_H_INCLUDED
