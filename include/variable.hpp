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

    namespace ceras_private
    {
        template< Tensor Tsor >
        struct session;
    }

    template< Tensor Tsor >
    ceras_private::session<Tsor>& get_default_session();

    template< Tensor Tsor >
    struct variable_state
    {
        Tsor data_;
        Tsor gradient_;
        std::vector<Tsor> contexts_;
    };

    template< typename Float > requires std::floating_point<Float>
    struct regularizer
    {
        typedef Float value_type;
        value_type l1_;
        value_type l2_;
        bool synchronized_;

        constexpr regularizer( value_type l1=0.0, value_type l2=0.0, bool synchronized=false ) noexcept : l1_{l1}, l2_{l2}, synchronized_{synchronized} {}
    };

    template< Tensor Tsor >
    struct variable : enable_id<variable<Tsor>, "Variable">
    {
        typedef Tsor tensor_type;
        typedef typename tensor_type::value_type value_type;

        std::shared_ptr<variable_state<tensor_type>> state_;
        regularizer<value_type> regularizer_;
        bool trainable_;

        variable( tensor_type const& data, value_type l1=value_type{0}, value_type l2=value_type{0}, bool trainable=true ) : enable_id<variable<tensor_type>, "Variable">{}, regularizer_{l1, l2, true}, trainable_{trainable}
        {
            (*this).state_ = std::make_shared<variable_state<tensor_type>>();
            (*((*this).state_)).data_ = data;
            (*((*this).state_)).gradient_ = tensor_type{ data.shape() };

            auto& ss = get_default_session<tensor_type>();
            ss.remember( *this );
        }

        //variable() = delete;
        variable() noexcept {}
        variable( variable const& other ) = default;
        variable( variable && ) = default;
        variable& operator=( variable&&) = default;
        variable& operator=( variable const& other) = default;

        tensor_type const forward() noexcept// const
        {
            auto& state = *((*this).state_);

            if ( learning_phase == 1 )
            {
                typedef typename tensor_type::value_type value_type;
                state.gradient_.reset( value_type{0} );
                regularizer_.synchronized_ = false; // mark changes
            }
            return state.data_;
        }

        void backward( auto const& grad ) noexcept
        {
            if (!trainable_) return;

            auto& state = *((*this).state_);
            {
                if (state.gradient_.shape() != state.data_.shape())
                    state.gradient_.resize( state.data_.shape() );
            }
            state.gradient_ += grad; // collecting all the gradients from its children nodes, will be called mulitple times in a single backward pass

            // apply regularizers
            if (!(regularizer_.synchronized_)) // in case of multiple invoke of this method in a same backward pass
            {
                if ( regularizer_.l1_ >= eps ) // l1 regularizer
                {
                    value_type const factor = regularizer_.l1_;
                    for_each( state.data_.begin(), state.data_.end(), state.gradient_.begin(), [factor]( value_type d, value_type& g ){ g += (d >= value_type{0}) ? factor : -factor; } );
                }
                if ( regularizer_.l2_ >= eps ) // l2 regularizer
                {
                    value_type const factor = regularizer_.l2_;
                    for_each( state.data_.begin(), state.data_.end(), state.gradient_.begin(), [factor]( value_type d, value_type& g ){ g += value_type{2} * d * factor; } );
                }

                regularizer_.synchronized_ = true;
            }
        }

        std::vector<std::size_t> shape() const noexcept
        {
            auto& state = *((*this).state_);
            //debug_log( fmt::format("calculating the shape of variable with id {}, got {}", (*this).id(), state.data_.shape() ) );
            return state.data_.shape();
        }

        std::vector<tensor_type>& contexts()
        {
            auto& state = *((*this).state_);
            return state.contexts_;
        }

        std::vector<tensor_type> contexts() const
        {
            auto& state = *((*this).state_);
            return state.contexts_;
        }

        tensor_type& data()
        {
            auto& state = *((*this).state_);
            return state.data_;
        }

        tensor_type data() const
        {
            auto& state = *((*this).state_);
            return state.data_;
        }

        tensor_type& gradient()
        {
            auto& state = *((*this).state_);
            return state.gradient_;
        }

        tensor_type gradient() const
        {
            auto& state = *((*this).state_);
            return state.gradient_;
        }

        void reset()
        {
            data().reset();
            gradient().reset();
        }

        /*
        void reset_states()
        {
            if ( stateful_ )
                reset();
        }
        */

        bool trainable() const noexcept { return trainable_; }
        void trainable( bool t ) { trainable_ = t; }
        /*
        bool stateful() const noexcept { return stateful_; }
        void stateful( bool s ){ stateful_ = s; }
        */

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

