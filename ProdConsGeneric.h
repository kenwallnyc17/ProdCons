#ifndef PRODCONSGENERIC_H_INCLUDED
#define PRODCONSGENERIC_H_INCLUDED

#include "Useful.h"

namespace PRODCONSGENERIC
{

/// to do:


///

template<typename T>
concept IsNTCopyMoveConstructible = (is_nothrow_copy_constructible_v<T> && is_nothrow_move_constructible_v<T>);

template<typename T>
concept IsNTMoveAssignable = (is_nothrow_move_assignable_v<T>);

template<typename T>
concept ProdConsRequires = (IsNTCopyMoveConstructible<T> && IsNTMoveAssignable<T>);

template<typename T>
concept IsNTDefConstructible = ( is_nothrow_default_constructible_v<T>);

///T must be nothrow move assignable, move constructible and copy constructible
template<ProdConsRequires T, size_t N>
class ProdConsGeneric
{
public:
    ProdConsGeneric() = default;
    virtual ~ProdConsGeneric(){}

    ProdConsGeneric(const ProdConsGeneric&) = delete;
    ProdConsGeneric& operator=(const ProdConsGeneric&) = delete;
    ProdConsGeneric(ProdConsGeneric&&) = delete;
    ProdConsGeneric& operator=(ProdConsGeneric&&) = delete;

    ///produce
    bool prod(const T& t) noexcept;
    bool prod(T&& t) noexcept;

    template<typename First, typename... Args>
    bool emplace(First&& fst, Args&&... rgs) noexcept(false);

    ///consume
    virtual bool cons(T& t) noexcept = 0;

protected:

    /// must either move the pointed to object or return false without moving it
    virtual bool emplace_impl(T*) noexcept = 0;

    virtual void write(const size_t idx, T&& t) noexcept = 0;
};

/// called for lvalue t
template<ProdConsRequires T, size_t N>
bool ProdConsGeneric<T,N>::prod(const T& t) noexcept
{
    return emplace(move(T(t)));/// move a copy
}

/// only called for rvalue t
template<ProdConsRequires T, size_t N>
bool ProdConsGeneric<T,N>::prod(T&& t) noexcept
{
    return emplace(forward<T>(t));
}

template<ProdConsRequires T, size_t N>
template<typename First, typename... Args>
bool ProdConsGeneric<T,N>::emplace(First&& fst, Args&&... rgs) noexcept(false)
{
    bool res{};

    /// if fst is an rvalue T, move it directly, otherwise, first create a new T object and then move it
    if constexpr(sizeof...(Args) == 0 && is_same_v<remove_reference_t<First>, T> && is_rvalue_reference_v<decltype(fst)>)
    {
        res = emplace_impl(&fst);
    }
    else
    {
        T* pt = new T(forward<First>(fst), forward<Args>(rgs)...);

        if((res = emplace_impl(pt)) == false)
           delete pt;
    }

    return res;
}


template<typename T, size_t N>
class ProdConsAlignedSlotArray : public ProdConsGeneric<T,N>
{
public:
    ProdConsAlignedSlotArray() noexcept(false);
    ~ProdConsAlignedSlotArray();

    ProdConsAlignedSlotArray(const ProdConsAlignedSlotArray&) = delete;
    ProdConsAlignedSlotArray& operator=(const ProdConsAlignedSlotArray&) = delete;
    ProdConsAlignedSlotArray(ProdConsAlignedSlotArray&&) = delete;
    ProdConsAlignedSlotArray& operator=(ProdConsAlignedSlotArray&&) = delete;

protected:

    void write(const size_t idx, T&& t) noexcept override;

    struct alignas(cacheline_size_bytes) Slot
    {
        alignas(cacheline_size_bytes) atomic<size_t> wr_rd{};/// wr if even (empty), rd if odd (full)
        alignas(cacheline_size_bytes) char storage[sizeof(T)];///
    };

    Slot* buf_{};
    const size_t sz_{powof2(N, cacheline_size_bytes)};
    const size_t buf_mask_{sz_-1};
};


template<typename T, size_t N>
ProdConsAlignedSlotArray<T,N>::ProdConsAlignedSlotArray() noexcept(false)
{
    /// want nothrow default ctor -- only propagate up call stack for bad alloc
    buf_ = new (std::align_val_t(page_size_bytes)) Slot[sz_];
}

template<typename T, size_t N>
ProdConsAlignedSlotArray<T,N>::~ProdConsAlignedSlotArray()
{
    if(buf_ == nullptr) return;

    for(size_t i = 0; i < sz_; ++i)
    {
        Slot& slt{buf_[i]};

        if(slt.wr_rd & 1)
        {
            reinterpret_cast<T*>(slt.storage)->~T();
        }
    }

    ::operator delete[](buf_, sz_*sizeof(Slot), std::align_val_t(page_size_bytes));
}

template<typename T, size_t N>
void ProdConsAlignedSlotArray<T,N>::write(const size_t, T&& ) noexcept
{

}


/// new base class for variable size data writes
/// N is requested buffer size in bytes, which is rounded up to be a powerof2
/// only support char buffer for now, maybe expand later, so keep templates
template<same_as<char> T, size_t N>
class ProdConsAlignedDataBuffer
{
public:
    ProdConsAlignedDataBuffer() noexcept(false);
    virtual ~ProdConsAlignedDataBuffer();

    ProdConsAlignedDataBuffer(const ProdConsAlignedDataBuffer&) = delete;
    ProdConsAlignedDataBuffer& operator=(const ProdConsAlignedDataBuffer&) = delete;
    ProdConsAlignedDataBuffer(ProdConsAlignedDataBuffer&&) = delete;
    ProdConsAlignedDataBuffer& operator=(ProdConsAlignedDataBuffer&&) = delete;

    ///user interface
    virtual bool prod(const char* p, uint32_t len) noexcept = 0;
    virtual void cons(char* p, uint32_t& len) noexcept = 0;/// len == 0 means nothing read

    size_t cap() const {return sz_;}

protected:

    /// for internal usage
    void write(const size_t idx, const char* p, const uint32_t len) noexcept;
    void read(const size_t idx, char* p, const uint32_t len) noexcept;

    T* buf_{};
    const size_t sz_{powof2(N, cacheline_size_bytes)};/// want this to be a divisible by cacheline size
    const size_t buf_mask_{sz_-1};

private:

};

template<same_as<char> T, size_t N>
ProdConsAlignedDataBuffer<T,N>::ProdConsAlignedDataBuffer() noexcept(false)
{
    buf_ = new (std::align_val_t(page_size_bytes)) T[sz_];
}

template<same_as<char> T, size_t N>
ProdConsAlignedDataBuffer<T,N>::~ProdConsAlignedDataBuffer()
{
  //  delete[] buf_;
   ::operator delete[](buf_, sz_*sizeof(T), std::align_val_t(page_size_bytes));
}

template<same_as<char> T, size_t N>
void ProdConsAlignedDataBuffer<T,N>::write(const size_t idx, const char* p, const uint32_t len) noexcept
{
    memcpy(&buf_[idx], p, len);
}

template<same_as<char> T, size_t N>
void ProdConsAlignedDataBuffer<T,N>::read(const size_t idx, char* p, const uint32_t len) noexcept
{
    memcpy(p, &buf_[idx], len);
}

/// writing variable sized data type is done via a page aligned char array
template<same_as<char> T, size_t N>
class ProdConsVariableData : public ProdConsAlignedDataBuffer<T,N>
{
public:
    ProdConsVariableData() = default;

    ProdConsVariableData(const ProdConsVariableData&) = delete;
    ProdConsVariableData& operator=(const ProdConsVariableData&) = delete;
    ProdConsVariableData(ProdConsVariableData&&) = delete;
    ProdConsVariableData& operator=(ProdConsVariableData&&) = delete;

    /// needs some type info in order to be generally useful
    template<typename U>
    bool prod(U&&) noexcept;/// write the variable sized type U (properly decayed) into the data buffer

protected:
    using ProdConsAlignedDataBuffer<T,N>::prod;
};


/// ?? use decltype(u) or just U
template<same_as<char> T, size_t N>
template<typename U>
[[gnu::flatten]]
bool ProdConsVariableData<T,N>::prod(U&& u) noexcept
{
  //  cout << "type of U, u = " << type_name<U>() << ", " << type_name<decltype(u)>() << endl;

    /// if array or function return false
    /// tests for detecting if is_array_v == true for these types declared outside of call to this func:
    ///     uint64_t iarr[3]{1};  YES -- is array
    ///     uint64_t iarr[] = {1,9,8};  YES -- is array
    ///     uint64_t* iarr{new uint64_t[5]{}}; NO -- is not an array
    if(is_array_v<remove_reference_t<decltype(u)>> || is_function_v<remove_reference_t<decltype(u)>>)
        return false;

    /// doesn't handle derived types if base pointer provided
    if constexpr(is_pointer_v<remove_reference_t<decltype(u)>>)
    {
        return prod(reinterpret_cast<const char*>(u), sizeof(remove_pointer_t<remove_reference_t<decltype(u)>>));
    }
    else
    {
       // cout << "ProdConsVariableData<T,N>::prod(U&& u) : " << reinterpret_cast<const char*>(&u) << ", " << sizeof(remove_reference_t<decltype(u)>) << endl;
        return prod(reinterpret_cast<const char*>(&u), sizeof(remove_reference_t<decltype(u)>));
    }
}


/// //////////////////////////////////////////////////////////////////////////////////////////////////////



}//PRODCONSGENERIC

#endif // PRODCONSGENERIC_H_INCLUDED
