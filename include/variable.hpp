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

    template< Tensor Tsor >
    struct variable : enable_id<variable<Tsor>>
    {
        typedef Tsor tensor_type;

        std::shared_ptr<variable_state<Tsor>> state_;
        bool trainable_;
        bool stateful_;

        variable( Tsor const& data, bool trainable = true, bool stateful = false ) : enable_id<variable<Tsor>>{}, trainable_{trainable}, stateful_{stateful}
        {
            (*this).state_ = std::make_shared<variable_state<Tsor>>();
            (*((*this).state_)).data_ = data;
            (*((*this).state_)).gradient_ = Tsor{ data.shape() };

            auto& ss = get_default_session<Tsor>();//.get();
            ss.remember( *this );
        }

        variable() = delete;
        variable( variable const& other ) = default;
        variable( variable && ) = default;
        variable& operator=( variable&&) = default;
        variable& operator=( variable const& other) = default;

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
            state.gradient_ += grad; // collecting all the gradients from its children nodes, will be called mulitple times in a single backward pass
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

        void reset()
        {
            data().reset();
            gradient().reset();
        }

        void reset_states()
        {
            if ( stateful_ )
                reset();
        }

        bool trainable() const noexcept { return trainable_; }
        void trainable( bool t ) { trainable_ = t; }
        bool stateful() const noexcept { return stateful_; }
        void stateful( bool s ){ stateful_ = s; }

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

