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

namespace ceras
{

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

        Loss&         loss_;
        //Loss          loss_;
        T             learning_rate_;
        T             momentum_;
        T             decay_;
        bool          nesterov_;
        unsigned long iterations_;

        sgd(Loss& loss, std::size_t batch_size, T learning_rate=1.0e-1, T momentum=0.0, T decay=0.0, bool nesterov=false) noexcept :
            loss_{loss}, learning_rate_(learning_rate), momentum_(std::max(T{0}, momentum)), decay_{std::max(T{0}, decay)}, nesterov_{nesterov}, iterations_{0}
        {
            better_assert( batch_size >= 1, "batch_size must be positive, but got: ", batch_size );
            learning_rate_ /= static_cast<T>( batch_size );
        }

        void forward()
        {
            loss_.backward( ones<T>( {1, } ) );
            learning_rate_ /= ( 1.0 + decay_ * iterations_ );
            auto& ss = get_default_session<tensor_type>();//.get();
            for ( auto [id, v] : ss.variables_ )
            {
                if (v.trainable_)
                {
                    auto& data = v.data();
                    auto& gradient = v.gradient();
                    auto& contexts = v.contexts();
                    if ( contexts.empty() ) // create context
                        contexts.push_back( zeros_like( data ) );
                    auto& moments = contexts[0];
                    for_each( moments.begin(), moments.end(), gradient.begin(), [this]( T& m, T g ) { m *= (*this).momentum_; m -= (*this).learning_rate_ * g;} );
                    if (!nesterov_ ) for_each( moments.begin(), moments.end(), data.begin(), gradient.begin(), [this]( T m, T& v, T g ) { v += (*this).momentum_ * m - (*this).learning_rate_ * g; } );
                    else data += moments;
                }
            }
            ++iterations_;
        }//sgd::forward
    };//sgd

    template< typename Loss, typename T >
    struct adagrad : enable_id<adagrad<Loss, T >, "adagrad optimizer">, enable_shared<adagrad<Loss,T>>
    {
        typedef tensor< T > tensor_type;

        Loss&         loss_;
        T             learning_rate_;
        T             decay_;
        unsigned long iterations_;

        adagrad(Loss& loss, std::size_t batch_size, T learning_rate=1.0e-1, T decay=0.0) noexcept :
                loss_(loss), learning_rate_(learning_rate), decay_{std::max(T{0}, decay)}, iterations_{0}
        {
            better_assert( batch_size >= 1, "batch_size must be positive, but got: ", batch_size );
            learning_rate_ /= static_cast<T>( batch_size );
        }

        void forward()
        {
            loss_.backward( ones<T>( {1, } ) );

            learning_rate_ /= ( 1.0 + decay_ * iterations_ );

            auto& ss = get_default_session<tensor_type>();//.get();
            for ( auto [id, v] : ss.variables_ )
            {
                if (v.trainable_)
                {
                    auto& data = v.data();
                    auto& gradient = v.gradient();
                    auto& contexts = v.contexts();
                    if ( contexts.empty() ) // create context
                        contexts.push_back( zeros_like( data ) );
                        //contexts.push_back( std::make_shared<tensor_type>( zeros_like( data ) ) );
                    auto& moments = contexts[0];

                    for_each( moments.begin(), moments.end(), gradient.begin(), []( T& m, T g ) { m  += g*g; } );

                    for_each( data.begin(), data.end(), gradient.begin(), moments.begin(), [this]( T& d, T g, T m ) { d -= (*this).learning_rate_ * g / (eps + std::sqrt(m)); } );
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

        Loss&         loss_;
        T             learning_rate_;
        T             rho_;
        T             decay_;
        unsigned long iterations_;

        rmsprop(Loss& loss, std::size_t batch_size, T learning_rate=1.0e-1, T rho=0.9, T decay=0.0) noexcept :
                loss_(loss), learning_rate_(learning_rate), rho_{rho},  decay_{std::max(T{0}, decay)}, iterations_{0}
        {
            better_assert( batch_size >= 1, "batch_size must be positive, but got: ", batch_size );
            learning_rate_ /= static_cast<T>( batch_size );
        }

        void forward()
        {
            loss_.backward( ones<T>( {1, } ) );

            learning_rate_ /= ( 1.0 + decay_ * iterations_ );

            auto& ss = get_default_session<tensor_type>();//.get();
            for ( auto [id, v] : ss.variables_ )
            {
                if (v.trainable_)
                {
                    auto& data = v.data();
                    auto& gradient = v.gradient();
                    auto& contexts = v.contexts();
                    if ( contexts.empty() ) // create context
                        contexts.push_back( zeros_like( data ) );
                        //contexts.push_back( std::make_shared<tensor_type>( zeros_like( data ) ) );
                    auto& moments = contexts[0];

                    if ( iterations_ == 0 )
                        for_each( moments.begin(), moments.end(), gradient.begin(), [this]( T& m, T g ) { m = g*g; } );
                    else
                        for_each( moments.begin(), moments.end(), gradient.begin(), [this]( T& m, T g ) { m *= (*this).rho_; m  += g*g*(1.0-(*this).rho_); } );

                    for_each( data.begin(), data.end(), gradient.begin(), moments.begin(), [this]( T& d, T g, T m ) { d -= (*this).learning_rate_ * g / (eps + std::sqrt(m)); } );
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

        Loss&         loss_;
        T             rho_;
        T             learning_rate_;
        unsigned long iterations_;

        adadelta(Loss& loss, std::size_t batch_size, T rho=0.9) noexcept : loss_(loss), rho_{rho}, iterations_{0}
        {
            better_assert( batch_size >= 1, "batch_size must be positive, but got: ", batch_size );
            learning_rate_ = T{1} / static_cast<T>( batch_size );
        }

        void forward()
        {
            loss_.backward( ones<T>( {1, } ) );

            auto& ss = get_default_session<tensor_type>();//.get();
            for ( auto [id, v] : ss.variables_ )
            {
                if (v.trainable_)
                {
                    auto& data = v.data();
                    auto& gradient = v.gradient();
                    auto& contexts = v.contexts();
                    if ( contexts.empty() ) // create context
                    {
                        //contexts.push_back( std::make_shared<tensor_type>( zeros_like( data ) ) );
                        //contexts.push_back( std::make_shared<tensor_type>( zeros_like( data ) ) );
                        contexts.push_back( zeros_like( data ) );
                        contexts.push_back( zeros_like( data ) );
                    }
                    auto& moments = contexts[0];
                    auto& delta = contexts[0];

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

        Loss&         loss_;
        T             learning_rate_;
        T             beta_1_;
        T             beta_2_;
        bool          amsgrad_;
        unsigned long iterations_;

        adam(Loss& loss, std::size_t batch_size, T learning_rate=1.0e-1, T beta_1=0.9, T beta_2=0.999, bool amsgrad=false) noexcept :
             loss_{loss}, learning_rate_{learning_rate}, beta_1_{beta_1}, beta_2_{beta_2}, amsgrad_{ amsgrad }, iterations_{0}
        {
            better_assert( batch_size >= 1, "batch_size must be positive, but got: ", batch_size );
            learning_rate_ /= static_cast<T>( batch_size );
        }

        void forward()
        {
            loss_.backward( ones<T>( {1, } ) );
            auto& ss = get_default_session<tensor_type>();//.get();
            for ( auto [id, v] : ss.variables_ )
            {
                if (v.trainable_)
                {
                    auto& data = v.data();
                    auto& gradient = v.gradient();
                    auto& contexts = v.contexts();
                    if ( contexts.empty() ) // create context
                    {
                        //contexts.push_back( std::make_shared<tensor_type>( zeros_like( data ) ) );
                        //contexts.push_back( std::make_shared<tensor_type>( zeros_like( data ) ) );
                        contexts.push_back( zeros_like( data ) );
                        contexts.push_back( zeros_like( data ) );
                    }
                    auto& m = contexts[0];
                    auto& v = contexts[1];

                    T const b_beta_1 = beta_1_;
                    T const b_beta_2 = beta_2_;

                    for_each( m.begin(), m.end(), gradient.begin(), [b_beta_1](T& m_, T g_){ m_ *= b_beta_1; m_ += g_*(1.0-b_beta_1); } );

                    for_each( v.begin(), v.end(), gradient.begin(), [b_beta_2](T& v_, T g_){ v_ *= b_beta_2; v_ += g_* g_*(1.0-b_beta_2); } );

                    T lr = learning_rate_ * std::sqrt( 1.0 - std::pow(beta_2_, iterations_+1) ) / ( 1.0 - std::pow(beta_1_, iterations_+1) );

                    if ( iterations_ > 1 )
                        for_each( data.begin(), data.end(), m.begin(), v.begin(), [lr]( T& d_, T m_, T v_ ){ d_ -= lr * m_ / (eps+std::sqrt(v_)); } );
                    else
                        for_each( data.begin(), data.end(), gradient.begin(), [this]( T& d_, T g_ ){ d_ -= (*this).learning_rate_ * g_; } );

                    // TODO: enabling amsgrad
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
        Loss& loss_;
        T learning_rate_;
        T momentum_;

        gradient_descent(Loss& loss, std::size_t batch_size, T learning_rate=1.0e-3, T momentum=0.0) noexcept : loss_(loss), learning_rate_(learning_rate), momentum_(momentum)
        {
            learning_rate_ /= static_cast<T>( batch_size ); // fix for batch size
        }

        void forward()
        {
            // update the gradient in the loss
            loss_.backward( ones<T>( {1, } ) );
            //update variables
            auto& ss = get_default_session<tensor_type>();//.get();
            for ( auto [id, v] : ss.variables_ )
            {
                if (v.trainable_)
                {
                    //v.data() -= learning_rate_ * (v.gradient());
                    better_assert( !has_nan(v.gradient()), "gradient_descent error, tensor with id ", id, " has a nan value." );
                    v.data() -= learning_rate_ * v.gradient();
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

