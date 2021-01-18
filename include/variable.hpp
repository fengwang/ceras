#ifndef QVETFVLYKDJJLDBPAMVBUWUGPWXIAIGMXUDVOFGQIHUHVOTBAWEMPJQEWJQIGGSTUCNDHLUYL
#define QVETFVLYKDJJLDBPAMVBUWUGPWXIAIGMXUDVOFGQIHUHVOTBAWEMPJQEWJQIGGSTUCNDHLUYL

#include "./includes.hpp"
#include "./tensor.hpp"
#include "./utils/id.hpp"
#include "./utils/debug.hpp"
#include "./config.hpp"

namespace ceras
{

    template< typename T, typename A >
    struct session;

    template< typename T, typename A >
    std::reference_wrapper<session<T, A>> get_default_session();


    template< typename T, typename A = default_allocator<T> >
    struct variable
    {
        int id_;
        std::shared_ptr<tensor<T, A>> data_;
        std::shared_ptr<tensor<T, A>> gradient_;
        std::shared_ptr<tensor<T, A>> old_gradient_;

        variable( tensor<T, A> const& data ) :
            id_{ generate_uid() },
            data_{ std::make_shared<tensor<T, A>>( data ) },
            gradient_{ std::make_shared<tensor<T, A>>(data.shape()) },
            old_gradient_{std::make_shared<tensor<T, A>>(data.shape())}
        { }
        variable() = delete;

        void backward( auto const& grad )
        {
            *gradient_ += grad; // collecting all the gradients from its children nodes, will be called mulitple times
            auto& ss = get_default_session<T, A>().get();
            ss.remember( *this );
        }

        tensor<T, A> const forward() const
        {
            if ( learning_phase == 1 )
            {
                std::swap( *gradient_, *old_gradient_ );
                (*gradient_).reset( T{0} );
            }
            return *data_;
        }

        std::vector<std::size_t> shape() const noexcept
        {
            return (*data_).shape();
        }
    };//struct variable

    template< typename T >
    struct is_variable : std::false_type {};

    template< typename T, typename A >
    struct is_variable< variable< T, A> > : std::true_type {};

    template< class T >
    inline constexpr bool is_variable_v = is_variable<T>::value;

    template< typename T >
    concept Variable = is_variable_v<T>;

}//namespace ceras

#endif//QVETFVLYKDJJLDBPAMVBUWUGPWXIAIGMXUDVOFGQIHUHVOTBAWEMPJQEWJQIGGSTUCNDHLUYL

