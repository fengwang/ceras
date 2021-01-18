#ifndef QVETFVLYKDJJLDBPAMVBUWUGPWXIAIGMXUDVOFGQIHUHVOTBAWEMPJQEWJQIGGSTUCNDHLUYL
#define QVETFVLYKDJJLDBPAMVBUWUGPWXIAIGMXUDVOFGQIHUHVOTBAWEMPJQEWJQIGGSTUCNDHLUYL

#include "./includes.hpp"
#include "./tensor.hpp"
#include "./utils/id.hpp"
#include "./utils/debug.hpp"
#include "./config.hpp"

namespace ceras
{

    template< Tensor Tsor >
    struct session;

    template< Tensor Tsor >
    std::reference_wrapper<session<Tsor>> get_default_session();


    template< Tensor Tsor >
    struct variable
    {
        int id_;
        std::shared_ptr<Tsor> data_;
        std::shared_ptr<Tsor> gradient_;
        std::shared_ptr<Tsor> old_gradient_;

        variable( Tsor const& data ) :
            id_{ generate_uid() },
            data_{ std::make_shared<Tsor>( data ) },
            gradient_{ std::make_shared<Tsor>(data.shape()) },
            old_gradient_{std::make_shared<Tsor>(data.shape())}
        { }
        variable() = delete;

        void backward( auto const& grad )
        {
            *gradient_ += grad; // collecting all the gradients from its children nodes, will be called mulitple times
            // TODO:
            auto& ss = get_default_session<Tsor>().get();
            ss.remember( *this );
        }

        Tsor const forward() const
        {
            if ( learning_phase == 1 )
            {
                typedef typename Tsor::value_type value_type;
                std::swap( *gradient_, *old_gradient_ );
                (*gradient_).reset( value_type{0} );
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

    template< Tensor Tsor >
    struct is_variable< variable<Tsor> > : std::true_type {};

    template< class T >
    inline constexpr bool is_variable_v = is_variable<T>::value;

    template< typename T >
    concept Variable = is_variable_v<T>;

}//namespace ceras

#endif//QVETFVLYKDJJLDBPAMVBUWUGPWXIAIGMXUDVOFGQIHUHVOTBAWEMPJQEWJQIGGSTUCNDHLUYL

