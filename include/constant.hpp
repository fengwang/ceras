#ifndef CONSTANT_HPP_INCLUDED_DLKJASLKJFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
#define CONSTANT_HPP_INCLUDED_DLKJASLKJFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

#include "./includes.hpp"
#include "./tensor.hpp"
#include "./utils/id.hpp"
#include "./utils/debug.hpp"

namespace ceras
{

    template< typename T, typename A = default_allocator<T> >
    struct constant
    {
        int id_;
        std::shared_ptr<tensor<T, A>> data_;

        constant( tensor<T, A> const& data ) : id_{ generate_uid() }, data_{ std::make_shared<tensor<T, A>>( data ) } { }
        constant() = delete;

        void backward( auto ) { }

        tensor<T, A> const forward() const
        {
            return *data_;
        }

        std::vector<std::size_t> shape() const noexcept
        {
            return (*data_).shape();
        }
    };//struct constant

    template< typename T >
    struct is_constant : std::false_type {};

    template< typename T, typename A >
    struct is_constant< constant< T, A> > : std::true_type {};

    template< class T >
    inline constexpr bool is_constant_v = is_constant<T>::value;

    template< typename T >
    concept Constant = is_constant_v<T>;

}//namespace ceras

#endif//CONSTANT_HPP_INCLUDED_DLKJASLKJFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

