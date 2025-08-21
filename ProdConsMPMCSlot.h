#ifndef PRODCONSMPMCSLOT_H_INCLUDED
#define PRODCONSMPMCSLOT_H_INCLUDED

#include "ProdConsGeneric.h"
#include "ProdConsTesting.h"

namespace PRODCONSSPECIFIC
{
using namespace PRODCONSGENERIC;


/**
    Comments on ProdConsMPMCSlot :

*/

template<typename T, size_t N, auto TOEXIT>
class ProdConsMPMCSlot final : public ProdConsAlignedSlotArray<T,N>
{
public:

    ProdConsMPMCSlot() noexcept = default;

    ProdConsMPMCSlot(const ProdConsMPMCSlot&) = delete;
    ProdConsMPMCSlot& operator=(const ProdConsMPMCSlot&) = delete;
    ProdConsMPMCSlot(ProdConsMPMCSlot&&) = delete;
    ProdConsMPMCSlot& operator=(ProdConsMPMCSlot&&) = delete;

    bool emplace_impl(T*) noexcept override;/// parent prod and emplace will eventually call this for specific behavior
    bool cons(T& t) noexcept override;

private:

    using ProdConsAlignedSlotArray<T,N>::buf_;
    using ProdConsAlignedSlotArray<T,N>::sz_;
    using ProdConsAlignedSlotArray<T,N>::buf_mask_;
    using typename ProdConsAlignedSlotArray<T,N>::Slot;

    /// global write and read indices (must be masked with buf_mask_ in order to get index into buf_)
    alignas(cacheline_size_bytes) atomic<size_t> widx_{}, ridx_{};
};


template<typename T, size_t N, auto TOEXIT>
[[gnu::flatten]]
bool ProdConsMPMCSlot<T,N,TOEXIT>::emplace_impl(T* pt) noexcept
{
    if(pt == nullptr) return false;

    const size_t my_widx{widx_.fetch_add(1, memory_order_acq_rel)};

    Slot& my_slot{buf_[my_widx & buf_mask_]};

    while(2*(my_widx/sz_) != my_slot.wr_rd.load(memory_order_acquire))
    {
        if(TOEXIT() == true)
            return false;
    }

    new (my_slot.storage) T(move(*pt));

    my_slot.wr_rd.store(2*(my_widx/sz_) + 1, memory_order_release);

    return true;
}


template<typename T, size_t N, auto TOEXIT>
[[gnu::flatten]]
bool ProdConsMPMCSlot<T,N,TOEXIT>::cons(T& t) noexcept
{
    const size_t my_ridx{ridx_.fetch_add(1, memory_order_acq_rel)};

    Slot& my_slot{buf_[my_ridx & buf_mask_]};

    while(2*(my_ridx/sz_) + 1 != my_slot.wr_rd.load(memory_order_acquire))
    {
        /// can get caught in this loop waiting indefinitely, so, allow for an exit request
        if(TOEXIT() == true)
            return false;
    }

    t = move(*reinterpret_cast<T*>(my_slot.storage));

    reinterpret_cast<T*>(my_slot.storage)->~T();

    my_slot.wr_rd.store(2*(my_ridx/sz_) + 2, memory_order_release);

    return true;
}

void test_mpmcslot_1()
{
    cout << "\nPRODCONSSPECIFIC::test_mpmcslot_1 \n" << endl;

    {
        constexpr size_t NUM2{20};
        PRODCONSGENERIC::ProdConsTester<ProdConsMPMCSlot, PRODCONSGENERIC::D1, NUM2, 2, 2> pct2;

        deque<PRODCONSGENERIC::D1> ds2;
        for(size_t i{}; i < 5*NUM2; ++i)
        {
            ds2.emplace_back(PRODCONSGENERIC::D1{-1,i});
        }

        pct2(ds2);
    }

    cout << "\n end PRODCONSSPECIFIC::test_mpmcslot_1 \n" << endl;
}


}

#endif // PRODCONSMPMCSLOT_H_INCLUDED
