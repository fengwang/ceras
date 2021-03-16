#ifndef OHWSAKJKEDNRMFSMIPCYSSOAHVYBOEXVLKQRYWAABVVDAOPPIGXPNCYSVMCIKLKECRVJFDWRH
#define OHWSAKJKEDNRMFSMIPCYSSOAHVYBOEXVLKQRYWAABVVDAOPPIGXPNCYSVMCIKLKECRVJFDWRH

#include "../includes.hpp"

namespace ceras
{

    /*
    template <class ContainerType>
    concept Container = requires(ContainerType a, const ContainerType b)
    {
        requires std::regular<ContainerType>;
        requires std::swappable<ContainerType>;
        requires std::destructible<typename ContainerType::value_type>;
        requires std::same_as<typename ContainerType::reference, typename ContainerType::value_type &>;
        requires std::same_as<typename ContainerType::const_reference, const typename ContainerType::value_type &>;
        requires std::forward_iterator<typename ContainerType::iterator>;
        requires std::forward_iterator<typename ContainerType::const_iterator>;
        requires std::signed_integral<typename ContainerType::difference_type>;
        requires std::same_as<typename ContainerType::difference_type, typename std::iterator_traits<typename ContainerType_::iterator>::difference_type>;
        requires std::same_as<typename ContainerType::difference_type, typename std::iterator_traits<typename ContainerType_::const_iterator>::difference_type>;
        { a.begin() } -> typename ContainerType::iterator;
        { a.end() } -> typename ContainerType::iterator;
        { b.begin() } -> typename ContainerType::const_iterator;
        { b.end() } -> typename ContainerType::const_iterator;
        { a.cbegin() } -> typename ContainerType::const_iterator;
        { a.cend() } -> typename ContainerType::const_iterator;
        { a.size() } -> typename ContainerType::size_type;
        { a.max_size() } -> typename ContainerType::size_type;
        { a.empty() } -> boolean;
    };
    */


    template <typename T>
    struct reversion_wrapper
    {
        T& iterable;
    };

    template <typename T>
    constexpr auto begin ( reversion_wrapper<T> w )
    {
        return std::rbegin( w.iterable );
    }

    template <typename T>
    constexpr auto end ( reversion_wrapper<T> w )
    {
        return std::rend( w.iterable );
    }

    template <typename T>
    constexpr reversion_wrapper<T> reverse ( T&& iterable )
    {
        return { iterable };
    }

}//namespace ceras
#endif//OHWSAKJKEDNRMFSMIPCYSSOAHVYBOEXVLKQRYWAABVVDAOPPIGXPNCYSVMCIKLKECRVJFDWRH

