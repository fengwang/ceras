#ifndef DJDWJBHNDAYTNOXLFOBDSGAQAAYPWMXJGEBYIRKEAKAQUUWVGDUGGDKSDXUKSPCYYNTWTDNII
#define DJDWJBHNDAYTNOXLFOBDSGAQAAYPWMXJGEBYIRKEAKAQUUWVGDUGGDKSDXUKSPCYYNTWTDNII

#include "./operation.hpp"
#include "./tensor.hpp"
#include "./utils/range.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/for_each.hpp"
#include "./utils/context_cast.hpp"

namespace ceras
{

    ///
    /// @brief Step activation function, an unary operator.
    ///
    /// @param ex An input operator
    ///
    /// \code{.cpp}
    /// auto x = Input();
    /// auto y = Dense( 10, 28*28 )( x );
    /// auto output = heaviside_step( y );
    /// \endcode
    ///
    template< std::floating_point Float >
    auto constexpr heaviside_step( Float f ) noexcept // f should not be zero
    {
        return [=]<Expression Ex>( Ex const& ex ) noexcept
        {
            return sigmoid( value( f+f ) * ex );
        };
    }

    // alias of heaviside_step(20)
    template <Expression Ex>
    auto constexpr soft_sign( Ex const& ex ) noexcept // soft-sign
    {
        return heaviside_step( 20.0 )( ex );
    }


    // alias of heaviside_step(20)
    template <Expression Ex>
    auto constexpr unit_step( Ex const& ex ) noexcept
    {
        return soft_sign( ex );
    }

    // alias of heaviside_step(20)
    template <Expression Ex>
    auto constexpr binary_step( Ex const& ex ) noexcept
    {
        return soft_sign( ex );
    }


    ///
    /// @brief Gaussian activation function, an unary operator.
    ///
    /// @param ex An input operator
    ///
    /// \code{.cpp}
    /// auto x = Input();
    /// auto y = Dense( 10, 28*28 )( x );
    /// auto output = gaussian( y );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr gaussian( Ex const& ex ) noexcept
    {
        return exp( negative( square(ex) ) );
    }





    ///
    /// @brief Softmax activation function, an unary operator.
    ///
    /// @param ex An input operator
    ///
    /// \code{.cpp}
    /// auto x = Input();
    /// auto y = Dense( 10, 28*28 )( x );
    /// auto output = softmax( y );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr softmax( Ex const& ex ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        better_assert( !input.empty(), "softmax forward: input tensor is empty!" );
                                        Tsor x = deep_copy( input );
                                        std::size_t const last_dim = *(x.shape().rbegin());
                                        std::size_t const rest_dim = x.size() / last_dim;
                                        for ( auto idx : range( rest_dim ) )
                                        {
                                            auto [begin, end] = std::make_tuple( x.begin()+idx*last_dim, x.begin()+(idx+1)*last_dim );
                                            typename Tsor::value_type const mx = *std::max_element( begin, end );
                                            for_each( begin, end, [mx]( auto & v ){ v = std::exp( v-mx ); } );
                                            typename Tsor::value_type const sum = std::accumulate( begin, end, typename Tsor::value_type{0} );
                                            for_each( begin, end, [sum]( auto & v ){ v /= sum; } );
                                        }
                                        return x;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "backprop: upcoming gradient for activation softmax contains NaN" );
                                        Tsor ans = grad;
                                        for_each( ans.begin(), ans.end(), output.begin(), []( auto& a, auto o ) { a *= o * ( typename Tsor::value_type{1} - o ); } );
                                        return ans;
                                    },
                                    "Softmax"
                )( ex );
    }

    ///
    /// @brief Scaled Exponential Linear Unit (SELU) activation function, an unary operator. If `x>0`, returns 1.0507 x; Otherwise, returns 1.67326*1.0507*(exp(x)-1)
    ///
    /// @param ex An input operator
    ///
    /// \code{.cpp}
    /// auto x = Input();
    /// auto y = Dense( 10, 28*28 )( x );
    /// auto output = selu( y );
    /// \endcode
    ///
    template <Expression Ex>
    auto inline selu( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        typedef typename Tsor::value_type value_type;
                                        value_type const lambda = 1.0507;
                                        value_type const alpha = 1.67326;
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        std::copy( input.begin(), input.end(), ans.begin() );
                                        // if x >= 0:  \lambda x
                                        // if x <  0:  \lambda \alpha (exp(x) - 1)
                                        ans.map( [lambda, alpha](auto& x){ x = (x >= value_type{0}) ? (lambda * x) : (lambda * alpha * (std::exp(x) - value_type{1})); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        typedef typename Tsor::value_type value_type;
                                        value_type const lambda = 1.0507;
                                        value_type const alpha = 1.67326;
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() ); // 1 / ( 1 + exp(-x) )
                                        // if x >= 0: \lambda
                                        // if x <  0: \lambda \alpha exp( x )
                                        for_each( ans.begin(), ans.end(), input.begin(), grad.begin(), [lambda, alpha]( auto& a, auto i, auto g ){ a = (i >= value_type{0}) ? (g * lambda) : (g * lambda * alpha * std::exp(i)); } );
                                        return ans;
                                    },
                                    "SeLU"
                )( ex );
    }

    ///
    /// @brief Softplus function, an unary operator. Returns `log(exp(x)+1)`.
    ///
    /// @param ex An input operator
    ///
    /// \code{.cpp}
    /// auto x = Input();
    /// auto y = Dense( 10, 28*28 )( x );
    /// auto output = softplus( y );
    /// \endcode
    ///
    template <Expression Ex>
    auto inline softplus( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        std::copy( input.begin(), input.end(), ans.begin() );
                                        ans.map( [](auto& x){ x = std::log(1.0+std::exp(x)); } ); // ln( 1+e^x )
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() ); // 1 / ( 1 + exp(-x) )
                                        for_each( ans.begin(), ans.end(), input.begin(), grad.begin(), []( auto& a, auto i, auto g ){ a = g / ( typename Tsor::value_type{1} - std::exp(-i) ); } );
                                        return ans;
                                    },
                                    "SoftPlus"
                )( ex );
    }


    ///
    /// @brief Softsign function, an unary operator. Returns ` x / (abs(x) + 1)`.
    ///
    /// @param ex An input operator.
    ///
    /// \code{.cpp}
    /// auto x = Input();
    /// auto y = Dense( 10, 28*28 )( x );
    /// auto output = softsign( y );
    /// \endcode
    ///
    template <Expression Ex>
    auto inline softsign( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        std::copy( input.begin(), input.end(), ans.begin() );
                                        ans.map( [](auto& x){ x /= typename Tsor::value_type{1} + std::abs(x); } ); //  x / ( 1+|x| )
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() ); // 1 / ( 1 + |x| )^2
                                        for_each( ans.begin(), ans.end(), input.begin(), grad.begin(), []( auto& a, auto i, auto g ){ auto tmp = typename Tsor::value_type{1} + std::abs(i); a = g / (tmp*tmp); } );
                                        return ans;
                                    },
                                    "SoftSign"
                )( ex );
    }

    ///
    /// @brief Sigmoid function, an unary operator. Returns `1 / (exp(-x) + 1)`.
    ///
    /// @param ex An input operator.
    ///
    /// \code{.cpp}
    /// auto x = Input();
    /// auto y = Dense( 10, 28*28 )( x );
    /// auto output = sigmoid( y );
    /// \endcode
    ///
    template <Expression Ex>
    auto inline sigmoid( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        std::copy( input.begin(), input.end(), ans.begin() );
                                        //auto ans = input.deep_copy();
                                        ans.map( [](auto& x){ x = 1.0 / (1.0+std::exp(-x)); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const&, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( output.shape() );
                                        //Tsor ans{ output.shape() };
                                        for_each( ans.begin(), ans.end(), output.begin(), grad.begin(), []( auto & a, auto o, auto g ){ a = g * o * ( typename Tsor::value_type{1} - o ); } );
                                        return ans;
                                    },
                                    "Sigmoid"
                )( ex );
    }


    namespace
    {
        struct relu_context
        {
            auto make_forward() const noexcept
            {
                return []( std::shared_ptr<std::any> forward_cache ) noexcept
                {
                    return [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                    {
                        typedef typename Tsor::value_type value_type;
                        Tsor& ans = context_cast<Tsor>( forward_cache );
                        ans.resize( input.shape()  );

                        for_each( ans.begin(), ans.end(), input.begin(), [](auto& o, auto x){ o = std::max(x, value_type{0}); } );

                        return ans;
                    };
                };
            }

            auto make_backward() const noexcept
            {
                return []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                {
                    typedef typename Tsor::value_type value_type;
                    Tsor ans = grad; // shallow copy
                    for_each( ans.begin(), ans.end(), input.begin(), []( auto& v, auto x ){ if ( x <= value_type{0} ) v = value_type{0}; } );
                    return ans;
                };
            }
        }; // relu_context

    }//anonymous namespace

    ///
    /// @brief Relu function, an unary operator. Returns `x` if positive, `0` otherwise.
    ///
    /// @param ex An input operator.
    ///
    /// \code{.cpp}
    /// auto x = Input();
    /// auto y = Dense( 10, 28*28 )( x );
    /// auto output = relu( y );
    /// \endcode
    ///
    template <Expression Ex>
    auto relu( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        return make_unary_operator( relu_context{}.make_forward()( forward_cache ), relu_context{}.make_backward(), "Relu")( ex );
    }


    namespace
    {
        struct relu6_context
        {
            auto make_forward() const noexcept
            {
                return []( std::shared_ptr<std::any> forward_cache ) noexcept
                {
                    return [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                    {
                        typedef typename Tsor::value_type value_type;
                        Tsor& ans = context_cast<Tsor>( forward_cache );
                        ans.resize( input.shape()  );
                        for_each( ans.begin(), ans.end(), input.begin(), [](auto& o, auto x){ o = std::min( value_type{6}, std::max(x, value_type{0}) ); } );
                        return ans;
                    };
                };
            }

            auto make_backward() const noexcept
            {
                return []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                {
                    typedef typename Tsor::value_type value_type;
                    Tsor ans = grad; // shallow copy
                    //const typename Tsor::value_type zero{0};
                    for_each( ans.begin(), ans.end(), input.begin(), []( auto& v, auto x ){ if ( (x <= value_type{0}) || (x >= value_type{6}) ) v = value_type{0}; } );
                    return ans;
                };
            }
        }; // relu6_context

    }//anonymous namespace

    ///
    /// @brief Rectified Linear 6 function, an unary operator. Returns `min(max(features, 0), 6)`.
    ///
    /// @param ex An input operator.
    ///
    /// \code{.cpp}
    /// auto x = Input();
    /// auto y = Dense( 10, 28*28 )( x );
    /// auto output = relu6( y );
    /// \endcode
    ///
    template <Expression Ex>
    auto relu6( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        return make_unary_operator( relu6_context{}.make_forward()( forward_cache ), relu6_context{}.make_backward(), "Relu6")( ex );
    }


    ///
    /// @brief Leaky Rectified Linear function, an unary operator. Returns `x` if positive, `alpha x` otherwise. `alpha` defaults to 0.2.
    ///
    /// @param ex An input operator.
    ///
    /// \code{.cpp}
    /// auto x = Input();
    /// auto y = Dense( 10, 28*28 )( x );
    /// auto output = leaky_relu(0.1f)( y );
    /// \endcode
    ///
    template< typename T > requires std::floating_point<T>
    auto leaky_relu( T const factor=0.2 ) noexcept
    {
        better_assert( factor > T{0}, "Expecting leak_relu with a factor greater than 0, but got factor = ", factor );
        better_assert( factor < T{1}, "Expecting leak_relu with a factor less than 1, but got factor = ", factor );
        return [factor]<Expression Ex>( Ex const& ex ) noexcept
        {
            std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
            return make_unary_operator( [factor, forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                        {
                                            Tsor& ans = context_cast<Tsor>( forward_cache );
                                            ans.resize( input.shape()  );
                                            for_each( ans.begin(), ans.end(), input.begin(), [factor]( auto& v_out, auto v_in ){ v_out = std::max( T{v_in}, T{factor*v_in} ); } );
                                            return ans;
                                        },
                                        [factor]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                        {
                                            typedef typename Tsor::value_type value_type;
                                            Tsor ans = grad;// OK for shallow copy
                                            for_each( ans.begin(), ans.end(), input.begin(), [factor]( value_type& v_back, value_type const v_in ){ v_back = (v_in > value_type{0}) ? v_back : factor*v_back; } );
                                            return ans;
                                        },
                                        "LeakyRelu"
                    )( ex );
        };
    }

    ///
    /// @PReLU is an alias name of Leaky_ReLU
    ///
    template< typename T > requires std::floating_point<T>
    auto prelu( T const factor ) noexcept
    {
        return leaky_relu( factor );
    }

    template <Expression Ex>
    auto negative_relu( Ex const& ex ) noexcept
    {
        return negative( relu( ex ) );
    }


    ///
    /// @brief Exponential Linear function, an unary operator. Returns `x` if positive, `alpha* (exp(x)-1)` otherwise. `alpha` defaults to 0.2.
    ///
    /// @param ex An input operator.
    ///
    /// \code{.cpp}
    /// auto x = Input();
    /// auto y = Dense( 10, 28*28 )( x );
    /// auto output = elu(0.1f)( y );
    /// \endcode
    ///
    template< typename T=float > requires std::floating_point<T>
    auto elu( T const alpha=1.0 ) noexcept
    {
        return [alpha]<Expression Ex>( Ex const& ex ) noexcept
        {
            std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
            return make_unary_operator( [alpha, forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                        {
                                            typedef typename Tsor::value_type value_type;
                                            Tsor& ans = context_cast<Tsor>( forward_cache );
                                            ans.resize( input.shape()  );
                                            for_each( ans.begin(), ans.end(), input.begin(), [alpha]( auto& v_out, auto v_in ){ v_out = (v_in > value_type{0}) ? v_in : (alpha * (std::exp(v_in) - value_type{1})); } );
                                            return ans;
                                        },
                                        [alpha]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                        {
                                            typedef typename Tsor::value_type value_type;
                                            Tsor ans = grad;// OK for shallow copy
                                            for_each( ans.begin(), ans.end(), input.begin(), [alpha]( value_type& v_back, value_type const v_in ){ v_back = (v_in >= value_type{0}) ? v_back : alpha*std::exp(v_back); } );
                                            return ans;
                                        },
                                        "ELU"
                    )( ex );
        };
    }

    ///
    /// @brief Exponential function, an unary operator. Returns `exp(x)`.
    ///
    /// @param ex An input operator.
    ///
    /// \code{.cpp}
    /// auto x = Input();
    /// auto y = Dense( 10, 28*28 )( x );
    /// auto output = exponential( y );
    /// \endcode
    ///
    template <Expression Ex>
    auto inline exponential( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();

        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        std::copy( input.begin(), input.end(), ans.begin() );
                                        ans.map( [](auto& x){ x = std::exp(x); } ); // exp(x)
                                        better_assert( !has_nan( ans ), "exponential operator forward output contains nan." );
                                        better_assert( !has_inf( ans ), "exponential operator forward output contains inf." );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        Tsor ans = grad;
                                        for_each( ans.begin(), ans.end(), output.begin(), []( auto& a, auto o ){ a *= o; } );
                                        return ans;
                                    },
                                    "Exponentional"
                )( ex );
    }

    ///
    /// @brief Hard Sigmoid function, an unary operator. Piecewise linear approximation of the sigmoid function.
    ///
    /// @param ex An input operator.
    ///
    /// \code{.cpp}
    /// auto x = Input();
    /// auto y = Dense( 10, 28*28 )( x );
    /// auto output = hard_sigmoid( y );
    /// \endcode
    ///
    template <Expression Ex>
    auto inline hard_sigmoid( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();

        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        typedef typename Tsor::value_type value_type;
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        std::copy( input.begin(), input.end(), ans.begin() );
                                        ans.map([](auto& x) { x = ( x > value_type{1} )  ? value_type{1} : ( x < value_type{-1} ) ? value_type{0} : (x+value_type{1})/value_type{2}; });
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        typedef typename Tsor::value_type value_type;
                                        Tsor ans = grad;
                                        for_each( ans.begin(), ans.end(), input.begin(), []( auto& a, auto x ) { a = ((x > value_type{1}) || (x < value_type{-1})) ? value_type{0} : (a / value_type{2}); } );
                                        return ans;
                                    },
                                    "HardSigmoid"
                )( ex );
    }

    ///
    /// @brief Gaussian Error function, an unary operator.
    /// GAUSSIAN ERROR LINEAR UNITS (GELUS) https://arxiv.org/pdf/1606.08415.pdf
    /// $f(x) = 0.5x (1 + tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$
    /// $df = x ( 1 + tanh[\sqrt{2/\pi}(x + 0.044715x^3)] ) +  \sqrt(2/\pi) x sech^2[\sqrt(2/\pi) x (1+0.44715x^2) (1+0.134145x^2) ]$
    /// where $sec^2(x) = 1 - tanh^2(x)$
    /// derivative generated using service from https://www.symbolab.com/solver/derivative-calculator
    ///
    /// @param ex An input operator.
    ///
    /// \code{.cpp}
    /// auto x = Input();
    /// auto y = Dense( 10, 28*28 )( x );
    /// auto output = gelu( y );
    /// \endcode
    ///
    template <Expression Ex>
    auto inline gelu( Ex const& ex ) noexcept
    {
        auto _gelu = []<typename T>( T x )
        {
            auto const ans = 0.5 * x * ( 1.0 + std::tanh( 0.79788456080286535588 * x ( 1.0 + 0.044715*x*x ) ) );
            return static_cast<T>( ans );
        };
        auto sech_2 = []( auto x )
        {
            return 1.0 - std::pow( std::tanh(x), 2 );
        };
        auto _dgelu = [sech_2]<typename T>( T x )
        {
            auto const sq_2_pi_x = 0.79788456080286535588 * x;
            auto const _xx = x * x;
            auto const ans = 0.5 * ( 1.0 + std::tanh( sq_2_pi_x * ( 1.0 + 0.044715 * _xx ) ) ) + sq_2_pi_x * sech_2( sq_2_pi_x * (1.0 + 0.044715 * _xx ) * ( 1.0 + 0.134145 * _xx) );
            return static_cast<T>( ans );
        };

        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();

        return make_unary_operator( [forward_cache, _gelu]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        //typedef typename Tsor::value_type value_type;
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        std::copy( input.begin(), input.end(), ans.begin() );
                                        ans.map([_gelu](auto& x) { x = _gelu(x); });
                                        return ans;
                                    },
                                    [_dgelu]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        //typedef typename Tsor::value_type value_type;
                                        Tsor ans = grad;
                                        for_each( ans.begin(), ans.end(), [&_dgelu]( auto& x ) {  x = _dgelu(x); } );
                                        return ans;
                                    },
                                    "GeLU"
                )( ex );
    }


    ///
    /// @brief Swish activation function.
    ///
    /// Reference: Ramachandran, Prajit, Barret Zoph, and Quoc V. Le. “Searching for Activation Functions.” ArXiv:1710.05941 [Cs], October 16, 2017. http://arxiv.org/abs/1710.05941.
    ///
    /// @param ex Input expression.
    ///
    template< Expression Ex >
    auto swish( Ex const& ex ) noexcept
    {
        return hadamard_product( ex, sigmoid( ex ) );
    }

    ///
    /// @brief An alias name of activation \link #swish.
    ///
    template< Expression Ex >
    auto silu( Ex const& ex ) noexcept
    {
        return swish( ex );
    }

    ///
    /// @brief Concatenated Rectified Linear Units, an activation function which preserves both positive and negative phase information while enforcing non-saturated non-linearity.
    ///
    /// Reference: Shang, Wenling, Kihyuk Sohn, Diogo Almeida, and Honglak Lee. “Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units.” ArXiv:1603.05201 [Cs], July 19, 2016. http://arxiv.org/abs/1603.05201.
    ///
    /// \code{.cpp}
    /// auto v = variable{ random<float>{ 3, 3 } };
    /// auto c = crelu( v );
    /// \endcode
    ///
    template< Expression Ex >
    auto crelu( Ex const& ex ) noexcept
    {
        return concatenate(-1)( relu(ex), relu(-ex) );
    }

    ///
    /// @brief Tank shrink function.
    ///
    /// \code{.cpp}
    /// auto v = variable{ random<float>{ 3, 3 } };
    /// auto c = tank_shrink( v );
    /// \endcode
    ///
    template< Expression Ex >
    auto tank_shrink( Ex const& ex ) noexcept
    {
        return ex - tanh( ex );
    }


    ///
    /// @brief Mish function.
    ///
    /// \code{.cpp}
    /// auto v = variable{ random<float>{ 3, 3 } };
    /// auto c = mish( v );
    /// \endcode
    ///
    template< Expression Ex >
    auto mish( Ex const& ex ) noexcept
    {
        return ex*tanh(softplus(ex));
    }


    ///
    /// @brief Lisht function.
    ///
    /// \code{.cpp}
    /// auto v = variable{ random<float>{ 3, 3 } };
    /// auto c = lisht( v );
    /// \endcode
    ///
    template< Expression Ex >
    auto lisht( Ex const& ex ) noexcept
    {
        return ex*tanh(ex);
    }

}//namespace ceras

#endif//DJDWJBHNDAYTNOXLFOBDSGAQAAYPWMXJGEBYIRKEAKAQUUWVGDUGGDKSDXUKSPCYYNTWTDNII

