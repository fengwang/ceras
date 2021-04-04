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
    struct constant : enable_shared<constant<Tsor>>
    {
        typedef Tsor tensor_type;

        std::shared_ptr<Tsor> data_;

        constant() = delete;
        constant( Tsor const& data ) : data_{ std::make_shared<Tsor>( data ) } {}
        constant( constant const& ) = default;
        constant( constant && ) = default;

        void backward( auto const& ) { }

        Tsor const forward() const
        {
            better_assert( data_, "Error: data is empty." );
            return *data_;
        }

        std::vector<std::size_t> shape() const noexcept
        {
            return (*data_).shape();
        }
    };//struct constant

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

