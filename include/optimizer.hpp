#ifndef XNRPSJMCYFXBDGNRJAWDNDIYQNGNXMRVLEHGNQWILKMTHGNOVHODLLXCCNIMUUFQSMOIYHDUD
#define XNRPSJMCYFXBDGNRJAWDNDIYQNGNXMRVLEHGNQWILKMTHGNOVHODLLXCCNIMUUFQSMOIYHDUD

#include "./config.hpp"
#include "./operation.hpp"
#include "./place_holder.hpp"
#include "./variable.hpp"
#include "./session.hpp"
#include "./utils/color.hpp"
#include "./utils/debug.hpp"
#include "./utils/id.hpp"
#include "./utils/enable_shared.hpp"
#include "./utils/fmt.hpp"

namespace ceras
{
    template<class Tsor> struct optimizer_parameter_state {
        Tsor first,second,maximum;
        unsigned long steps=0;
        optimizer_parameter_state()=default;
        optimizer_parameter_state(optimizer_parameter_state const& s):first(s.first.empty()?Tsor{}:s.first.deep_copy()),second(s.second.empty()?Tsor{}:s.second.deep_copy()),maximum(s.maximum.empty()?Tsor{}:s.maximum.deep_copy()),steps(s.steps) {}
        optimizer_parameter_state& operator=(optimizer_parameter_state const& s) {if(this!=&s){auto copy=s;*this=std::move(copy);}return *this;}
        optimizer_parameter_state(optimizer_parameter_state&&)=default;
        optimizer_parameter_state& operator=(optimizer_parameter_state&&)=default;
    };
    template<class Ex, class F>
    void visit_parameters(Ex& ex, F const& f) {
        if constexpr (is_variable_v<Ex>) f(ex);
        else if constexpr (is_unary_operator_v<Ex>) visit_parameters(ex.op(),f);
        else if constexpr (is_binary_operator_v<Ex>) {
            visit_parameters(ex.lhs_op(),f); visit_parameters(ex.rhs_op(),f);
        }
    }
    template<class Tsor, class Ex>
    auto parameters_of(Ex& ex) {
        std::unordered_map<int, variable<Tsor>> parameters;
        visit_parameters(ex,[&](auto& v){parameters.insert_or_assign(v.id(),v);});
        return parameters;
    }


    // sgd:
    //     - loss:
    //     - batch_size:
    //     - learning_rate:
    //     - momentum:
    //     - decay: should be very small, such as 1.0e-8
    //     - nesterov:
    //
    template< typename Loss, typename T >
    struct sgd : enable_id<sgd<Loss, T>, "sgd optimizer">, enable_shared<sgd<Loss, T>>
    {
        typedef tensor< T > tensor_type;
        static constexpr bool is_optimizer=true;
        using parameter_state=optimizer_parameter_state<tensor_type>;
        std::unordered_map<int, parameter_state> states_;
        void reset_state(){states_.clear();iterations_=0;}

        Loss          loss_;
        T             learning_rate_;
        T             momentum_;
        T             decay_;
        bool          nesterov_;
        unsigned long iterations_;

        sgd(Loss& loss, std::size_t batch_size, T learning_rate=1.0e-1, T momentum=0.0, T decay=0.0, bool nesterov=false) :
            loss_{loss}, learning_rate_(learning_rate), momentum_(std::max(T{0}, momentum)), decay_{std::max(T{0}, decay)}, nesterov_{nesterov}, iterations_{0}
        {
            if(!batch_size) throw std::invalid_argument("batch size must be positive");
            // Loss reduction already normalizes its gradient.
        }

        void forward()
        {
            loss_.backward( ones<T>( {1, } ) );
            T const rate=learning_rate_ / ( 1.0 + decay_ * iterations_ );
            for ( auto [id, v] : parameters_of<tensor_type>(loss_) )
            {
                if (v.trainable())
                {
                    auto& data = v.data();
                    auto& gradient = v.gradient();
                    auto& state = states_[id];
                    if (state.first.empty() || state.first.shape()!=data.shape()) {
                        state.first=zeros_like(data); state.second=zeros_like(data);state.maximum=zeros_like(data);state.steps=0;
                    }
                    auto& moments = state.first;
                    for_each( moments.begin(), moments.end(), gradient.begin(), [this,rate]( T& m, T g ) { m *= (*this).momentum_; m -= rate * g;} );
                    if (nesterov_ ) for_each( moments.begin(), moments.end(), data.begin(), gradient.begin(), [this,rate]( T m, T& v, T g ) { v += (*this).momentum_ * m - rate * g; } );
                    else data += moments;

                    gradient.reset(); // clear variable gradient
                }
            }
            ++iterations_;
        }//sgd::forward
    };//sgd

    template< typename Loss, typename T >
    struct adagrad : enable_id<adagrad<Loss, T >, "adagrad optimizer">, enable_shared<adagrad<Loss,T>>
    {
        typedef tensor< T > tensor_type;
        static constexpr bool is_optimizer=true;
        using parameter_state=optimizer_parameter_state<tensor_type>;
        std::unordered_map<int, parameter_state> states_;
        void reset_state(){states_.clear();iterations_=0;}

        Loss          loss_;
        T             learning_rate_;
        T             decay_;
        unsigned long iterations_;

        adagrad(Loss& loss, std::size_t batch_size, T learning_rate=1.0e-1, T decay=0.0) :
                loss_(loss), learning_rate_(learning_rate), decay_{std::max(T{0}, decay)}, iterations_{0}
        {
            if(!batch_size) throw std::invalid_argument("batch size must be positive");
            // Loss reduction already normalizes its gradient.
        }

        void forward()
        {
            loss_.backward( ones<T>( {1, } ) );

            T const rate=learning_rate_ / ( 1.0 + decay_ * iterations_ );

            for ( auto [id, v] : parameters_of<tensor_type>(loss_) )
            {
                if (v.trainable())
                {
                    auto& data = v.data();
                    auto& gradient = v.gradient();
                    auto& state = states_[id];
                    if (state.first.empty() || state.first.shape()!=data.shape()) {
                        state.first=zeros_like(data); state.second=zeros_like(data);state.maximum=zeros_like(data);state.steps=0;
                    }
                    auto& moments = state.first;

                    for_each( moments.begin(), moments.end(), gradient.begin(), []( T& m, T g ) { m  += g*g; } );

                    for_each( data.begin(), data.end(), gradient.begin(), moments.begin(), [this,rate]( T& d, T g, T m ) { d -= rate * g / (eps + std::sqrt(m)); } );

                    gradient.reset(); // clear variable gradient
                }
            }
            ++iterations_;
        }//forward
    };//adagrad

    template< typename Loss, typename T >
    using ada_grad = adagrad<Loss, T>;

    template< typename Loss, typename T >
    struct rmsprop : enable_id< rmsprop< Loss, T >, "rmsprop optimizer" >, enable_shared<rmsprop<Loss, T>>
    {
        typedef tensor< T > tensor_type;
        static constexpr bool is_optimizer=true;
        using parameter_state=optimizer_parameter_state<tensor_type>;
        std::unordered_map<int, parameter_state> states_;
        void reset_state(){states_.clear();iterations_=0;}

        Loss          loss_;
        T             learning_rate_;
        T             rho_;
        T             decay_;
        unsigned long iterations_;

        rmsprop(Loss& loss, std::size_t batch_size, T learning_rate=1.0e-1, T rho=0.9, T decay=0.0) :
                loss_(loss), learning_rate_(learning_rate), rho_{rho},  decay_{std::max(T{0}, decay)}, iterations_{0}
        {
            if(!batch_size) throw std::invalid_argument("batch size must be positive");
            // Loss reduction already normalizes its gradient.
        }

        void forward()
        {
            loss_.backward( ones<T>( {1, } ) );

            T const rate=learning_rate_ / ( 1.0 + decay_ * iterations_ );

            for ( auto [id, v] : parameters_of<tensor_type>(loss_) )
            {
                if (v.trainable())
                {
                    auto& data = v.data();
                    auto& gradient = v.gradient();
                    auto& state = states_[id];
                    if (state.first.empty() || state.first.shape()!=data.shape()) {
                        state.first=zeros_like(data); state.second=zeros_like(data);state.maximum=zeros_like(data);state.steps=0;
                    }
                    auto& moments = state.first;

                    for_each(moments.begin(),moments.end(),gradient.begin(),[this](T& m,T g){m=rho_*m+(1-rho_)*g*g;});

                    for_each( data.begin(), data.end(), gradient.begin(), moments.begin(), [this,rate]( T& d, T g, T m ) { d -= rate * g / (eps + std::sqrt(m)); } );

                    gradient.reset(); // clear variable gradient
                }
            }
            ++iterations_;
        }//forward
    };//rmsprop

    template< typename Loss, typename T >
    using rms_prop = rmsprop< Loss, T >;

    template< typename Loss, typename T >
    struct adadelta : enable_id< adadelta< Loss, T >, "adadelta optimizer" >, enable_shared<adadelta<Loss, T>>
    {
        typedef tensor< T > tensor_type;
        static constexpr bool is_optimizer=true;
        using parameter_state=optimizer_parameter_state<tensor_type>;
        std::unordered_map<int, parameter_state> states_;
        void reset_state(){states_.clear();iterations_=0;}

        Loss          loss_;
        T             rho_;
        T             learning_rate_;
        unsigned long iterations_;

        adadelta(Loss& loss, std::size_t batch_size, T rho=0.9) : loss_(loss), rho_{rho}, iterations_{0}
        {
            if(!batch_size) throw std::invalid_argument("batch size must be positive");
            learning_rate_ = T{1};
        }

        void forward()
        {
            loss_.backward( ones<T>( {1, } ) );

            for ( auto [id, v] : parameters_of<tensor_type>(loss_) )
            {
                if (v.trainable())
                {
                    auto& data = v.data();
                    auto& gradient = v.gradient();
                    auto& state = states_[id];
                    if (state.first.empty() || state.first.shape()!=data.shape()) {
                        state.first=zeros_like(data); state.second=zeros_like(data);state.maximum=zeros_like(data);state.steps=0;
                    }
                    auto& moments = state.first;
                    auto& delta = state.second;

                    /*
                    if (iterations_==0)
                    {
                        for_each( moments.begin(), moments.end(), gradient.begin(), []( T& m, T g ) { m += g*g; } );
                        for_each( delta.begin(), delta.end(), gradient.begin(), []( T& d, T g ) { d += g*g; } );
                    }
                    else
                    {
                        // m = rho * m + (1-rho) * g * g;
                        for_each( moments.begin(), moments.end(), gradient.begin(), [this]( T& m, T g ) { m *= (*this).rho_; m  += g*g*(1.0-(*this).rho_); } );
                    }
                    */

                    for_each( moments.begin(), moments.end(), gradient.begin(), [this]( T& m, T g ) { m *= (*this).rho_; m  += g*g*(1.0-(*this).rho_); } );

                    // g_ = \sqrt{ (delta+eps) / (m+eps) }
                    for_each( gradient.begin(), gradient.end(), delta.begin(), moments.begin(), [this]( T& g, T d, T m ){ g *= (*this).learning_rate_ * std::sqrt((d+eps)/(m+eps));} );
                    // x = x - g_
                    data -= gradient;
                    // delta = rho * delta + (1-rho) * g_ * g_
                    /*
                    if (iterations_!=0)
                    */
                    for_each( delta.begin(), delta.end(), gradient.begin(), [this]( T& d, T g ) { d *= (*this).rho_; d += (1.0-(*this).rho_) * g * g; } );

                    gradient.reset(); // clear variable gradient
                }
            }
            ++iterations_;
        }//forward
    };//adadelta

    template< typename Loss, typename T >
    using ada_delta = adadelta< Loss, T >;

    template< typename Loss, typename T >
    struct adam : enable_id< adam< Loss, T >, "adam optimizer" >, enable_shared<adam<Loss, T>>
    {
        typedef tensor< T > tensor_type;
        static constexpr bool is_optimizer=true;
        using parameter_state=optimizer_parameter_state<tensor_type>;
        std::unordered_map<int, parameter_state> states_;
        void reset_state(){states_.clear();iterations_=0;}

        Loss          loss_;
        T             learning_rate_;
        T             beta_1_;
        T             beta_2_;
        bool          amsgrad_;
        unsigned long iterations_;

        adam(Loss& loss, std::size_t batch_size, T learning_rate=1.0e-1, T beta_1=0.9, T beta_2=0.999, bool amsgrad=false) :
             loss_{loss}, learning_rate_{learning_rate}, beta_1_{beta_1}, beta_2_{beta_2}, amsgrad_{ amsgrad }, iterations_{0}
        {
            if(!batch_size) throw std::invalid_argument("batch size must be positive");
            // Loss reduction already normalizes its gradient.
        }

        void forward()
        {
            loss_.backward( ones<T>( {1, } ) );
            for ( auto [id, v] : parameters_of<tensor_type>(loss_) )
            {
                if (v.trainable())
                {
                    auto& data = v.data();
                    auto& gradient = v.gradient();
                    auto& state = states_[id];
                    if (state.first.empty() || state.first.shape()!=data.shape()) {
                        state.first=zeros_like(data); state.second=zeros_like(data);state.maximum=zeros_like(data);state.steps=0;
                    }
                    auto& m = state.first;
                    auto& v = state.second;

                    T const b_beta_1 = beta_1_;
                    T const b_beta_2 = beta_2_;

                    for_each( m.begin(), m.end(), gradient.begin(), [b_beta_1](T& m_, T g_){ m_ *= b_beta_1; m_ += g_*(1.0-b_beta_1); } );

                    for_each( v.begin(), v.end(), gradient.begin(), [b_beta_2](T& v_, T g_){ v_ *= b_beta_2; v_ += g_* g_*(1.0-b_beta_2); } );

                    ++state.steps;
                    T correction1=1-std::pow(beta_1_,state.steps),correction2=1-std::pow(beta_2_,state.steps);
                    for(std::size_t i=0;i<data.size();++i) {
                        T variance=v[i];
                        if(amsgrad_) {state.maximum[i]=std::max(state.maximum[i],variance);variance=state.maximum[i];}
                        data[i]-=learning_rate_*(m[i]/correction1)/(std::sqrt(variance/correction2)+eps);
                    }

                    gradient.reset(); // clear variable gradient
                }
            }//loop of variables
            ++iterations_;
        }//adam::forward
    };// adam



    // Example usage:
    //
    //  //session ss;
    //  auto& ss = get_default_session<tensor<float>>();
    //  auto loss = ...;
    //  auto optimizer = gradient{ loss, 1.0e-3f };
    //  for i = 1 : 1000
    //      ss.run( loss, batch_size )
    //      ss.run( optimizer )
    //
    template< typename Loss, typename T >
    struct gradient_descent : enable_id< gradient_descent< Loss, T >, "gradient_descent optimizer" >, enable_shared<gradient_descent<Loss, T>>
    {
        typedef tensor< T > tensor_type;
        static constexpr bool is_optimizer=true;
        using parameter_state=optimizer_parameter_state<tensor_type>;
        std::unordered_map<int, parameter_state> states_;
        void reset_state(){states_.clear();iterations_=0;}
        Loss loss_;
        T learning_rate_;
        T momentum_;
        unsigned long iterations_=0;

        gradient_descent(Loss& loss, std::size_t batch_size, T learning_rate=1.0e-3, T momentum=0.0) : loss_(loss), learning_rate_(learning_rate), momentum_(momentum)
        {
            if(!batch_size) throw std::invalid_argument("batch size must be positive");
        }

        void forward()
        {
            // update the gradient in the loss
            loss_.backward( ones<T>( {1, } ) );
            //update variables
            for ( auto [id, v] : parameters_of<tensor_type>(loss_) )
            {
                if (v.trainable())
                {
                    //v.data() -= learning_rate_ * (v.gradient());
                    //
                    auto& gradient = v.gradient();
                    better_assert( !has_nan(gradient), "gradient_descent error, tensor with id ", id, " has a nan value." );
                    auto& state=states_[id];
                    if(state.first.shape()!=v.data().shape()) state.first=zeros_like(v.data());
                    for(std::size_t i=0;i<gradient.size();++i) {
                        state.first[i]=momentum_*state.first[i]-learning_rate_*gradient[i];
                        v.data()[i]+=state.first[i];
                    }

                    gradient.reset(); // clear variable gradient
                }
            }
        }

    };

    // TODO: adamax, nadam, ftrl



    //
    // optimizers interfaces
    //

    inline auto Adam = []( auto ... args )
    {
        return [=]<Expression Ex>( Ex& loss )
        {
            return adam{loss, args...};
        };
    };

    inline auto SGD = []( auto ... args )
    {
        return [=]<Expression Ex>( Ex& loss )
        {
            return sgd{loss, args...};
        };
    };

    inline auto Adagrad = []( auto ... args )
    {
        return [=]<Expression Ex>( Ex& loss )
        {
            return adagrad{loss, args...};
        };
    };

    inline auto RMSprop = []( auto ... args )
    {
        return [=]<Expression Ex>( Ex& loss )
        {
            return rmsprop{loss, args...};
        };
    };

    inline auto Adadelta = []( auto ... args )
    {
        return [=]<Expression Ex>( Ex& loss )
        {
            return adadelta{loss, args...};
        };
    };


}//namespace ceras

#endif//XNRPSJMCYFXBDGNRJAWDNDIYQNGNXMRVLEHGNQWILKMTHGNOVHODLLXCCNIMUUFQSMOIYHDUD

