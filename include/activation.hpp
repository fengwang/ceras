#ifndef DJDWJBHNDAYTNOXLFOBDSGAQAAYPWMXJGEBYIRKEAKAQUUWVGDUGGDKSDXUKSPCYYNTWTDNII
#define DJDWJBHNDAYTNOXLFOBDSGAQAAYPWMXJGEBYIRKEAKAQUUWVGDUGGDKSDXUKSPCYYNTWTDNII

#include "./operation.hpp"
#include "./tensor.hpp"
#include "./utils/range.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/for_each.hpp"
#include "./utils/context_cast.hpp"

#if 0
+ relu function
+ sigmoid function
+ softmax function
+ tanh function
+ softplus function
+ softsign function
+ selu function
+ elu function
+ gelu function
exponential function
#endif


namespace ceras
{
    // TODO:
    // for expression/activation with only single input/output, the corresponding tensor can be reused without allocating new memory
    //

    template <Expression Ex>
    auto constexpr softmax( Ex const& ex ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        better_assert( !input.empty(), "softmax forward: input tensor is empty!" );
                                        debug_print( "softmax forwarding, input tensor has ", input.size(),  " elements." );
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
                                        //Tsor ans{ output.shape() };
                                        //for_each( ans.begin(), ans.end(), output.begin(), grad.begin(), []( auto & a, auto o, auto g ){ a = g * o * ( typename Tsor::value_type{1} - o ); } );
                                        Tsor ans = grad;
                                        for_each( ans.begin(), ans.end(), output.begin(), []( auto& a, auto o ) { a *= o * ( typename Tsor::value_type{1} - 0 ); } );
                                        return ans;
                                    },
                                    "Softmax"
                )( ex );
    }

    template <Expression Ex>
    auto inline selu( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        //TODO: optimize backward_cache out by reusing grad

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
                                    }
                )( ex );
    }

    template <Expression Ex>
    auto inline softplus( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        //TODO: optimize backward_cache out by reusing grad

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
                                    }
                )( ex );
    }

    template <Expression Ex>
    auto inline softsign( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        //TODO: optimize backward_cache out by reusing grad

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
                                    }
                )( ex );
    }

    template <Expression Ex>
    //auto constexpr sigmoid( Ex const& ex ) noexcept
    auto inline sigmoid( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        //TODO: optimize backward_cache out by reusing grad
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        debug_print( "Sigmoid operator forwarded with tensor ", input.id_ );
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

    template <Expression Ex>
    //auto constexpr tanh( Ex const& ex ) noexcept
    auto inline tanh( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        //TODO: optimize backward_cache out by reusing grad
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        std::copy( input.begin(), input.end(), ans.begin() );
                                        //Tsor ans = input.deep_copy();
                                        ans.map( [](auto& x){ x = 2.0 / (1.0+std::exp(-2.0*x)) - 1.0; } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const&, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( output.shape() );
                                        std::copy( output.begin(), output.end(), ans.begin() );
                                        //auto ans = output.deep_copy();
                                        ans.map( []( auto& x ){ x = typename Tsor::value_type{1} - x * x; } );
                                        ans *= grad;
                                        return ans;
                                    }
                )( ex );
    }

    template <Expression Ex>
    auto relu( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape()  );
                                        for ( auto idx : range( ans.size() ) ) // 1-D view of tensors input and ans
                                            ans[idx] = std::max( input[idx], typename Tsor::value_type{0} );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor ans = grad; // shallow copy
                                        const typename Tsor::value_type zero{0};
                                        for ( auto idx : range( ans.size() ) ) // 1-D view of tensors input, grad and ans
                                            ans[idx] = ( input[idx] > zero ) ? grad[idx] : zero;
                                        return ans;
                                    },
                                    "Relu"
                )( ex );
    }

    template< typename T > requires std::floating_point<T>
    auto leaky_relu( T const factor ) noexcept
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
                                            //Tsor ans{ input.shape() };
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
                                        "Leaky Relu"
                    )( ex );
        };
    }

    template< typename T > requires std::floating_point<T>
    auto elu( T const alpha ) noexcept
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
                                        }
                    )( ex );
        };
    }

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
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        Tsor ans = grad;
                                        for_each( ans.begin(), ans.end(), output.begin(), []( auto& a, auto o ){ a *= o; } );
                                        return ans;
                                    }
                )( ex );
    }

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
                                        ans.map([](auto& x)
                                                {
                                                    x = ( x > value_type{1} )  ? value_type{1} :
                                                        ( x < value_type{-1} ) ? value_type{0} :
                                                        (x+value_type{1})/value_type{2};
                                                });
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        typedef typename Tsor::value_type value_type;
                                        Tsor ans = grad;
                                        for_each( ans.begin(), ans.end(), input.begin(), []( auto& a, auto x ) { a = ((x > value_type{1}) || (x < value_type{-1})) ? value_type{0} : (a / value_type{2}); } );
                                        return ans;
                                    }
                )( ex );
    }

    // GAUSSIAN ERROR LINEAR UNITS (GELUS) https://arxiv.org/pdf/1606.08415.pdf
    // f(x) = 0.5x (1 + tanh[\sqrt{2/π}(x + 0.044715x^3)])
    // df = x ( 1 + tanh[\sqrt{2/π}(x + 0.044715x^3)] ) +  \sqrt(2/π) x sech^2[\sqrt(2/π) x (1+0.44715x^2) (1+0.134145x^2) ]
    // where sec^2(x) = 1 - tanh^2(x)
    // derivative generated using service from https://www.symbolab.com/solver/derivative-calculator
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
                                    }
                )( ex );
    }

}//namespace ceras

#endif//DJDWJBHNDAYTNOXLFOBDSGAQAAYPWMXJGEBYIRKEAKAQUUWVGDUGGDKSDXUKSPCYYNTWTDNII

