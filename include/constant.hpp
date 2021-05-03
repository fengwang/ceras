#ifndef CONSTANT_HPP_INCLUDED_DLKJASLKJFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
#define CONSTANT_HPP_INCLUDED_DLKJASLKJFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

#include "./includes.hpp"
#include "./tensor.hpp"
#include "./utils/id.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/enable_shared.hpp"

namespace ceras
{

    template< Tensor Tsor >
    struct constant
    {
        // Tsor is a shallow copy, and once an instance is initialized,it will never change
        Tsor data_;

        constant( Tsor const& data ) : data_{data} {}

        void backward( auto ) const {}

        Tsor forward() const
        {
            return data_;
        }

        auto shape() const
        {
            return data_.shape();
        }
    };

    template< typename T >
    struct is_constant : std::false_type {};

    template< Tensor Tsor >
    struct is_constant< constant< Tsor > > : std::true_type {};

    template< class T >
    inline constexpr bool is_constant_v = is_constant<T>::value;

    template< typename T >
    concept Constant = is_constant_v<T>;

}//namespace ceras

#endif//CONSTANT_HPP_INCLUDED_DLKJASLKJFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

