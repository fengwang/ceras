#ifndef QVETFVLYKDJJLDBPAMVBUWUGPWXIAIGMXUDVOFGQIHUHVOTBAWEMPJQEWJQIGGSTUCNDHLUYL
#define QVETFVLYKDJJLDBPAMVBUWUGPWXIAIGMXUDVOFGQIHUHVOTBAWEMPJQEWJQIGGSTUCNDHLUYL

#include "./includes.hpp"
#include "./tensor.hpp"
#include "./utils/id.hpp"
#include "./utils/debug.hpp"
#include "./config.hpp"
#include "./utils/enable_shared.hpp"
#include "./utils/state.hpp"
#include "./utils/type2string.hpp"
#include "./utils/fmt.hpp"

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
        std::vector<Tsor> contexts_; // compatibility only; optimizers own their state
        bool trainable_=true;
        typename Tsor::value_type l1_=0, l2_=0;
        bool synchronized_=true;
    };

    template< typename Float > requires std::floating_point<Float>
    struct regularizer
    {
        typedef Float value_type;
        value_type l1_;
        value_type l2_;
        bool synchronized_;

        constexpr regularizer( value_type l1=0.0, value_type l2=0.0, bool synchronized=false ) : l1_{l1}, l2_{l2}, synchronized_{synchronized} {}
    };

    template< Tensor Tsor >
    struct variable : enable_id<variable<Tsor>, "Variable">
    {
        typedef Tsor tensor_type;
        typedef typename tensor_type::value_type value_type;

        std::shared_ptr<variable_state<tensor_type>> state_;

        variable( tensor_type const& data, value_type l1=value_type{0}, value_type l2=value_type{0}, bool trainable=true ) : enable_id<variable<tensor_type>, "Variable">{}
        {
            (*this).state_ = std::make_shared<variable_state<tensor_type>>();
            state_->trainable_=trainable; state_->l1_=l1; state_->l2_=l2;
            (*((*this).state_)).data_ = data;
            (*((*this).state_)).gradient_ = tensor_type{ data.shape() };

            auto& ss = get_default_session<tensor_type>();
            ss.remember( *this );
        }

        //variable() = delete;
        variable() : state_{std::make_shared<variable_state<tensor_type>>()} {}
        variable( variable const& other ) = default;
        variable( variable && ) = default;
        variable& operator=( variable&&) = default;
        variable& operator=( variable const& other) = default;

        tensor_type const forward()// const
        {
            get_default_session<tensor_type>().remember(*this);
            auto& state = *((*this).state_);

            if ( learning_phase == 1 )
            {
                typedef typename tensor_type::value_type value_type;
                state.gradient_.reset( value_type{0} );
                state_->synchronized_ = false; // mark changes
            }
            return state.data_;
        }

        void backward( auto const& grad )
        {
            if (!state_->trainable_) return;

            auto& state = *((*this).state_);
            {
                if (state.gradient_.shape() != state.data_.shape())
                    state.gradient_.resize( state.data_.shape() );
            }
            state.gradient_ += grad; // collecting all the gradients from its children nodes, will be called mulitple times in a single backward pass

            // apply regularizers
            if (!(state_->synchronized_)) // in case of multiple invoke of this method in a same backward pass
            {
                if ( state_->l1_ >= eps ) // l1 regularizer
                {
                    value_type const factor = state_->l1_;
                    for_each( state.data_.begin(), state.data_.end(), state.gradient_.begin(), [factor]( value_type d, value_type& g ){ g += (d >= value_type{0}) ? factor : -factor; } );
                }
                if ( state_->l2_ >= eps ) // l2 regularizer
                {
                    value_type const factor = state_->l2_;
                    for_each( state.data_.begin(), state.data_.end(), state.gradient_.begin(), [factor]( value_type d, value_type& g ){ g += value_type{2} * d * factor; } );
                }

                state_->synchronized_ = true;
            }
        }

        std::vector<std::size_t> shape() const
        {
            auto& state = *((*this).state_);
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

        bool trainable() const { return state_->trainable_; }
        bool& trainable() { return state_->trainable_; }

        void trainable( bool t ) { state_->trainable_ = t; }

        value_type l1_regularizer() const
        {
            return state_->l1_;
        }

        value_type& l1_regularizer()
        {
            return state_->l1_;
        }

        value_type l2_regularizer() const
        {
            return state_->l2_;
        }

        value_type& l2_regularizer()
        {
            return state_->l2_;
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
    bool operator == ( Var const& lhs, Var const& rhs )
    {
        return lhs.id_ == rhs.id_;
    }

    template< Variable Var >
    std::tuple<std::string, std::vector<std::string>> const serialize( Var const& var )
    {
        auto const& [data_name, data_code] = serialize( var.data() );
        // serialize the gradient? TODO: usefule to store checkpoint for a training process

        std::string var_name = fmt::format( "variable_{}", var.id() );
        std::vector<std::string> var_code = data_code;
        //variable( tensor_type const& data, value_type l1=value_type{0}, value_type l2=value_type{0}, bool trainable=true ) : enable_id<variable<tensor_type>, "Variable">{}
        var_code.emplace_back( fmt::format( "ceras::variable<ceras::tensor<{}>> {}( {}/*tensor*/, {}/*l1 regularizer*/, {}/*l2 regularizer*/, {}/*trainable*/ );", type2string<typename Var::value_type>(), var_name, data_name, var.l1_regularizer(), var.l2_regularizer(), var.trainable()  ) );

        return std::forward_as_tuple( var_name, var_code );
    }


}//namespace ceras

#endif//QVETFVLYKDJJLDBPAMVBUWUGPWXIAIGMXUDVOFGQIHUHVOTBAWEMPJQEWJQIGGSTUCNDHLUYL

