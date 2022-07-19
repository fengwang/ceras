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
            return data();
        }

        auto shape() const
        {
            return data().shape();
        }

        tensor_type data() const noexcept
        {
            return data_;
        }

        tensor_type& data() noexcept
        {
            return data_;
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

    template< Constant Con >
    std::tuple<std::string, std::vector<std::string>> const serialize( Con const& con )
    {
        std::string constant_name = fmt::format( "constant_{}", con.id() );

        auto const& [tensor_name, tensor_code] = serialize( con.data() );

        std::vector<std::string> constant_code = tensor_code;
        constant_code.emplace_back( fmt::format( "ceras::constant<ceras::tensor<{}>> {} ( {} );", type2string<typename Con::value_type>(), constant_name, tensor_name ) );

        return std::forward_as_tuple(  constant_name, constant_code );
    }

}//namespace ceras

#endif//CONSTANT_HPP_INCLUDED_DLKJASLKJFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

