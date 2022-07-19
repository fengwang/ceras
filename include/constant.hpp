#ifndef CONSTANT_HPP_INCLUDED_DLKJASLKJFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
#define CONSTANT_HPP_INCLUDED_DLKJASLKJFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

#include "./includes.hpp"
#include "./tensor.hpp"
#include "./utils/id.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/enable_shared.hpp"
#include "./utils/type2string.hpp"

namespace ceras
{

    ///
    /// @brief Creates a constant expression from a tensor-like object.
    ///
    /// \code{.cpp}
    /// auto c = constant{ zeros<float>( {3, 3, 3} ) };
    /// \endcode
    ///
    template< Tensor Tsor >
    struct constant : enable_id<constant<Tsor>, "Constant">
    {
        typedef Tsor tensor_type;
        typedef typename tensor_type::value_type value_type;

        tensor_type data_;

        constant( tensor_type const& data ) : enable_id<constant<tensor_type>, "Constant">{}, data_{data} {}

        void backward( auto ) const {}

        tensor_type forward() const
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

#if 0
    template< Constant Con >
    std::tuple<std::string, std::vector<std::string>> serialize( Con const& con )
    {

    }
#endif

}//namespace ceras

#endif//CONSTANT_HPP_INCLUDED_DLKJASLKJFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

