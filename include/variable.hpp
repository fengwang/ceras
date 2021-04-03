#ifndef QVETFVLYKDJJLDBPAMVBUWUGPWXIAIGMXUDVOFGQIHUHVOTBAWEMPJQEWJQIGGSTUCNDHLUYL
#define QVETFVLYKDJJLDBPAMVBUWUGPWXIAIGMXUDVOFGQIHUHVOTBAWEMPJQEWJQIGGSTUCNDHLUYL

#include "./includes.hpp"
#include "./tensor.hpp"
#include "./utils/id.hpp"
#include "./utils/debug.hpp"
#include "./config.hpp"
#include "./utils/enable_shared.hpp"
#include "./utils/state.hpp"

namespace ceras
{

    template< Tensor Tsor >
    struct session;

    template< Tensor Tsor >
    std::reference_wrapper<session<Tsor>> get_default_session();

    template< Tensor Tsor >
    struct variable_state
    {
        Tsor data_;
        Tsor gradient_;
        std::vector<Tsor> contexts_;
    };

    template< Tensor Tsor >
    struct variable :   enable_id<variable<Tsor>>,
                        enable_shared_state<variable<Tsor>, variable_state<Tsor>>
    {
        bool trainable_;

        variable( Tsor const& data ) : trainable_{true}
        {
            (*this).state_ = std::make_shared<variable_state<Tsor>>();
            (*((*this).state_)).data_ = data;
            (*((*this).state_)).gradient_ = Tsor{ data.shape() };
        }
        variable() = delete;
        variable( variable const& ) = default;
        variable( variable && ) = default;
        variable& operator=( variable&&) = default;
        variable& operator=( variable const&) = default;

        Tsor const forward() const
        {
            auto& state = *((*this).state_);

            if ( learning_phase == 1 )
            {
                typedef typename Tsor::value_type value_type;
                state.gradient_.reset( value_type{0} );
            }
            return state.data_;
        }

        void backward( auto const& grad )
        {
            if (!trainable_) return;

            auto& state = *((*this).state_);
            state.gradient_ += grad; // collecting all the gradients from its children nodes, will be called mulitple times
            auto& ss = get_default_session<Tsor>().get();
            ss.remember( *this );
        }

        std::vector<std::size_t> shape() const noexcept
        {
            auto& state = *((*this).state_);
            return state.data_.shape();
        }

        std::vector<Tsor>& contexts()
        {
            auto& state = *((*this).state_);
            return state.contexts_;
        }

        std::vector<Tsor> contexts() const
        {
            auto& state = *((*this).state_);
            return state.contexts_;
        }

        Tsor& data()
        {
            auto& state = *((*this).state_);
            return state.data_;
        }

        Tsor data() const
        {
            auto& state = *((*this).state_);
            return state.data_;
        }

        Tsor& gradient()
        {
            auto& state = *((*this).state_);
            return state.gradient_;
        }

        Tsor gradient() const
        {
            auto& state = *((*this).state_);
            return state.gradient_;
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

    template< Variable Var >
    bool operator == ( Var const& lhs, Var const& rhs ) noexcept
    {
        return lhs.id_ == rhs.id_;
    }

}//namespace ceras

#endif//QVETFVLYKDJJLDBPAMVBUWUGPWXIAIGMXUDVOFGQIHUHVOTBAWEMPJQEWJQIGGSTUCNDHLUYL

