#ifndef XEAVBHOQHNWANMKWOLPWCMFDUQUSLURVORBSSSVDJOQDMREXBGTSRBLCPHUIOVUEEIIKYSRPK
#define XEAVBHOQHNWANMKWOLPWCMFDUQUSLURVORBSSSVDJOQDMREXBGTSRBLCPHUIOVUEEIIKYSRPK

#include "../includes.hpp"

namespace ceras
{
    namespace
    {
        template< typename TIter >
        struct iterator
        {
            long int i;
            TIter iter;

            constexpr auto operator <=>( iterator const& ) const noexcept = default;
            //constexpr bool operator != (const iterator & other) const { return iter != other.iter; }
            constexpr void operator ++ () noexcept { ++i; ++iter; }
            constexpr auto operator * () const noexcept { return std::tie(i, *iter); }
        }; //struct iterator

        template< typename T >
        struct iterable_wrapper
        {
            T iterable;

            constexpr auto begin() noexcept { return iterator{ 0L, std::begin(iterable) }; }
            constexpr auto end() noexcept { return iterator{ std::distance(std::begin(iterable), std::end(iterable)), std::end(iterable) }; }
        }; //struct iterable_wrapper

    }//anonymous namespace


    template <typename T, typename = decltype(std::begin(std::declval<T>())), typename = decltype(std::end(std::declval<T>()))>
    constexpr auto enumerate(T && iterable)
    {
        return iterable_wrapper{ std::forward<T>(iterable) };
    }

}//namespace ceras

#endif//XEAVBHOQHNWANMKWOLPWCMFDUQUSLURVORBSSSVDJOQDMREXBGTSRBLCPHUIOVUEEIIKYSRPK

