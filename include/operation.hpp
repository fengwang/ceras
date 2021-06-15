#ifndef IPKVWSJOCMGGVRASCBLPYHFBCHRIVEXYBOMMDAKFAUDFYVYOOOISLRXJNUJKPJEVMLDPRDSNM
#define IPKVWSJOCMGGVRASCBLPYHFBCHRIVEXYBOMMDAKFAUDFYVYOOOISLRXJNUJKPJEVMLDPRDSNM

#include "./includes.hpp"
#include "./place_holder.hpp"
#include "./variable.hpp"
#include "./constant.hpp"
#include "./value.hpp"
#include "./utils/range.hpp"
#include "./utils/debug.hpp"
#include "./config.hpp"
#include "./utils/context_cast.hpp"
#include "./utils/for_each.hpp"
#include "./utils/id.hpp"
#include "./utils/enable_shared.hpp"

namespace ceras
{
    //
    // an operator is composed of
    // 1. a left operator, a right operator and a lambda function, OR
    // 2. an operator and a lambda function
    //

    template< typename Operator, typename Forward_Action, typename Backward_Action >
    struct unary_operator : enable_id<unary_operator<Operator, Forward_Action, Backward_Action>, "Unary Operator">
    {
        Operator op_;
        Forward_Action forward_action_;
        Backward_Action backward_action_;

        typedef decltype( std::declval<Forward_Action>()( std::declval<decltype(op_)>().forward() ) ) tensor_type;

        tensor_type input_data_;
        tensor_type output_data_;

        unary_operator( Operator const& op, Forward_Action const& forward_action, Backward_Action const& backward_action ) noexcept :
            op_{op}, forward_action_{ forward_action }, backward_action_{ backward_action } { }

        auto forward()// const
        {
            input_data_ = op_.forward();
            output_data_ = forward_action_( input_data_ );
            return output_data_;
        }

        void backward( tensor_type const& grad )
        {
            auto const& current_gradient = backward_action_( input_data_, output_data_, grad );
            op_.backward( current_gradient );
        }

    };

    static auto constexpr make_unary_operator = []( auto const& unary_forward_action, auto const& unary_backward_action, std::string const& name="Anonymous Unary Operator" ) noexcept
    {
        return [&unary_forward_action, &unary_backward_action, &name]( auto const& op ) noexcept
        {
            auto ans = unary_operator{ op, unary_forward_action, unary_backward_action };
            ans.name_ = name;
            return ans;
        };
    };

    template< typename Lhs_Operator, typename Rhs_Operator, typename Forward_Action, typename Backward_Action >
    struct binary_operator :enable_id<binary_operator<Lhs_Operator, Rhs_Operator, Forward_Action, Backward_Action>, "Binary Operator">
    {
        Lhs_Operator lhs_op_;
        Rhs_Operator rhs_op_;
        Forward_Action forward_action_;
        Backward_Action backward_action_; // backward action for binary operator produces a tuple of two tensors

        typedef typename tensor_deduction<Lhs_Operator, Rhs_Operator>::tensor_type tensor_type; // defined in value.hpp

        tensor_type lhs_input_data_;
        tensor_type rhs_input_data_;
        tensor_type output_data_;

        binary_operator( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op, Forward_Action const& forward_action, Backward_Action const& backward_action ) noexcept :
            lhs_op_{lhs_op}, rhs_op_{rhs_op}, forward_action_{ forward_action }, backward_action_{ backward_action } { }

        auto forward()
        {
            static_assert( !(is_value_v<Lhs_Operator> && is_value_v<Rhs_Operator>), "Not valid for two values" );

            if constexpr ( is_value_v<Lhs_Operator> )
            {
                rhs_input_data_ = rhs_op_.forward();
                lhs_input_data_ = lhs_op_.forward( rhs_input_data_ );
            }
            else if constexpr ( is_value_v<Rhs_Operator> )
            {
                lhs_input_data_ = lhs_op_.forward();
                rhs_input_data_ = rhs_op_.forward( lhs_input_data_ );
            }
            else
            {
                lhs_input_data_ = lhs_op_.forward();
                rhs_input_data_ = rhs_op_.forward();
            }
            output_data_ = forward_action_( lhs_input_data_, rhs_input_data_ );
            return output_data_;
        }

        void backward( tensor_type const& grad )
        {
            auto const& [current_gradient_lhs, current_gradient_rhs] = backward_action_( lhs_input_data_, rhs_input_data_, output_data_, grad );
            lhs_op_.backward( current_gradient_lhs );
            rhs_op_.backward( current_gradient_rhs );
        }

    };

    static auto constexpr make_binary_operator = []( auto const& binary_forward_action, auto const& binary_backward_action, std::string const& name="Anonymous Binary Operator" ) noexcept
    {
        return [&binary_forward_action, &binary_backward_action, &name]( auto const& lhs_op, auto const& rhs_op ) noexcept
        {
            auto ans = binary_operator{ lhs_op, rhs_op, binary_forward_action, binary_backward_action };
            ans.name_ = name;
            return ans;
        };
    };

    template< typename T >
    struct is_unary_operator : std::false_type{};

    template< typename Operator, typename Forward_Action, typename Backward_Action >
    struct is_unary_operator< unary_operator<Operator, Forward_Action, Backward_Action> > : std::true_type {};

    ///
    /// If T is an instance of a unary_operator, the constant value equals to `true`. Otherwise this value is `false`.
    ///
    template< class T >
    inline constexpr bool is_unary_operator_v = is_unary_operator<T>::value;

    ///
    /// @concept Unary_Operator<>
    /// @brief A type that represents an unary operator.
    ///
    template< typename T >
    concept Unary_Operator = is_unary_operator_v<T>;


    template< typename T >
    struct is_binary_operator : std::false_type{};

    template< typename Lhs_Operator, typename Rhs_Operator, typename Forward_Action, typename Backward_Action >
    struct is_binary_operator< binary_operator<Lhs_Operator, Rhs_Operator, Forward_Action, Backward_Action> > : std::true_type {};

    ///
    /// If T is an instance of a binary_operator, the constant value equals to `true`. Otherwise this value is `false`.
    ///
    template< class T >
    inline constexpr bool is_binary_operator_v = is_binary_operator<T>::value;

    ///
    /// @concept Binary_Operator<>
    /// @brief A type that represents a binary operator.
    ///
    template< typename T >
    concept Binary_Operator = is_binary_operator_v<T>;

    ///
    /// @concept Operator<>
    /// @brief A type that represents an unary or a binary operator.
    ///
    template< typename T >
    concept Operator = Unary_Operator<T> || Binary_Operator<T>;

    ///
    /// @concept Expression<>
    /// @brief A type that represents a unary operator, a binary operator, a variable, a place_holder, a constant or a value
    ///
    template< typename T >
    concept Expression = Operator<T> || Variable<T> || Place_Holder<T> || Constant<T> || Value<T>;


    namespace
    {
        // `plus_context` was nested in the `plus` function.
        // But gcc compiler (11.1.0) has a lot of problems in deducing the context types (gcc may consume infnite memory and then get killed).
        // Moving forward/backward algorithms here to make gcc happy, at a price of an extra indirection layer with redundant code.
        struct plus_context
        {
            auto make_forward() const noexcept
            {
                return []<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
                {
                    better_assert( !has_nan( lhs_tensor ), "forward propagation for operator plus: lhs_tensor contains Nan!" );
                    better_assert( !has_nan( rhs_tensor ), "forward propagation for operator plus: rhs_tensor contains Nan!" );
                    return add( lhs_tensor, rhs_tensor );
                };
            }

            auto const make_backward() const noexcept
            {
                return []<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const& grad ) noexcept
                {
                   better_assert( !has_nan( grad ), "backprop: upcoming gradient for operator + contains NaN!" );

                   auto const& grad_fun = [&grad]( auto const& input )
                   {
                       Tsor ans = grad.deep_copy();
                       while( input.ndim() < ans.ndim() )
                           ans = sum( ans, 0 );
                       auto const& shape = input.shape();
                       for ( auto axis : range( input.ndim() ) )
                           if ( shape[axis] == 1 )
                              ans = sum( ans, axis, true );
                       return ans;
                   };
                   return std::make_tuple( grad_fun( lhs_input), grad_fun( rhs_input ) );
                };
            }
        }; // plus_context
    }//anonymous namespace

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr plus( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        plus_context const context_;
        return make_binary_operator( context_.make_forward(), context_.make_backward(), "Plus")( lhs_ex, rhs_ex );
    }

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr operator + ( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return plus( lhs_ex, rhs_ex );
    }

    namespace
    {
        struct multiplication_context
        {
            auto make_forward() const noexcept
            {
                return []( std::shared_ptr<std::any> forward_cache ) noexcept
                {
                    return [forward_cache]<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
                    {
                        Tsor& ans = context_cast<Tsor>( forward_cache );
                        multiply( lhs_tensor, rhs_tensor, ans );
                        return ans;
                        //return multiply( lhs_tensor, rhs_tensor );
                    };
                };
            }
            auto make_backward() const noexcept
            {
                return []<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const& grad ) noexcept
                {
                   // left branch <-- grad * rhs^T
                   auto const& g_shape = grad.shape();
                   auto const[m, n] = std::make_tuple( g_shape[0], g_shape[1] ); // 4, 1
                   auto const k = *(lhs_input.shape().rbegin()); // 13

                   Tsor lhs_grad{ lhs_input.shape() };

                   gemm( grad.data(), false, rhs_input.data(), true, m, n, k, lhs_grad.data() );

                   // right branch <-- lhs^T * grad
                   Tsor rhs_grad{ rhs_input.shape() };
                   gemm( lhs_input.data(), true, grad.data(), false, k, m, n, rhs_grad.data() );

                   return std::make_tuple( lhs_grad, rhs_grad );
                };
            }
        };//multiplication_context
    }//anonymous namespace

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto operator * ( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        // case of Value * Operator and Operator * Value
        if constexpr( is_value_v<Lhs_Expression> || is_value_v<Rhs_Expression> )
        {
            return elementwise_product( lhs_ex, rhs_ex );
        }
        else
        {
            multiplication_context const context_;
            std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
            return make_binary_operator( context_.make_forward()(forward_cache), context_.make_backward(), "Multiply")( lhs_ex, rhs_ex );
            //return make_binary_operator( context_.make_forward(), context_.make_backward(), "Multiply")( lhs_ex, rhs_ex );
        }
    }

    template <Expression Ex>
    auto constexpr log( Ex const& ex ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        better_assert( !has_nan( input ), "forward propagation for operator log: input contains Nan!" );
                                        auto ans = input.deep_copy();
                                        ans.map( [](auto & x){ better_assert( x+eps > 0, "log forward propagation, found an invalid value ", x ); x = std::log(x+eps); } );
                                        better_assert( !has_nan( ans ), "forward propagation for operator log: output contains Nan!" );
                                        better_assert( !has_inf( ans ), "forward propagation for operator log: output contains Inf!" );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator log contains NaN!" );
                                        auto ans = elementwise_divide(grad, input); // TODO: error here
                                        better_assert( !has_nan( ans ), "backprop: result for operator log contains NaN!" );
                                        return ans;
                                    }
                )( ex );
    };

    template <Expression Ex>
    auto constexpr negative( Ex const& ex ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& tensor ) noexcept
                                    {
                                        better_assert( !has_nan( tensor ), "forward propagation for operator log: tensor contains Nan!" );
                                        return -tensor;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator negative contains NaN!" );
                                        return -grad;
                                    }
                )( ex );
    };

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr elementwise_product( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return make_binary_operator( []<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
                                     {
                                        return elementwise_product( lhs_tensor, rhs_tensor );
                                     },
                                     []<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const grad ) noexcept
                                     {
                                        auto const& grad_fun = [&grad]( auto const& input, auto const& other_input )
                                        {
                                            Tsor ans = elementwise_product( grad, other_input );
                                            while( input.ndim() < ans.ndim() )
                                                ans = sum( ans, 0 );
                                            auto const& shape = input.shape();
                                            for ( auto axis : range( input.ndim() ) )
                                                if ( shape[axis] == 1 )
                                                    ans = sum( ans, axis, true );
                                            return ans;
                                        };
                                        return std::make_tuple( grad_fun( lhs_input, rhs_input ), grad_fun( rhs_input, lhs_input ) );
                                     }
                )( lhs_ex, rhs_ex );
    };

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr elementwise_multiply( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return elementwise_product( lhs_ex, rhs_ex );
    }

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr hadamard_product( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return elementwise_product( lhs_ex, rhs_ex );
    }

    template <Expression Ex>
    auto constexpr sum_reduce( Ex const& ex ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& tsor ) noexcept
                                    {
                                        better_assert( !has_nan( tsor ), "forward propagation for operator sum_reduce: tensor contains Nan!" );
                                        return reduce_sum( tsor );
                                    },
                                    []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator sum_reduce contains NaN!" );
                                        better_assert( grad.size() == 1, "sum_reduce should only output one value" );
                                        Tsor ans = ones_like( input );
                                        ans *= grad[0];
                                        return ans;
                                    }
                )( ex );
    }

    template <Expression Ex>
    auto constexpr reduce_sum( Ex const& ex ) noexcept
    {
        return sum_reduce( ex );
    }

    template <Expression Ex>
    auto constexpr mean_reduce( Ex const& ex ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& tsor ) noexcept
                                    {
                                        better_assert( !has_nan( tsor ), "forward propagation for operator mean: tensor contains Nan!" );
                                        return reduce_mean( tsor );
                                    },
                                    []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator mean_reduce contains NaN!" );
                                        better_assert( grad.size() == 1, "mean_reduce should only output one value" );
                                        Tsor ans = ones_like( input );
                                        ans *= grad[0];
                                        unsigned long const batch_size = (input.shape().size() == 1) ? 1 : (*(input.shape().begin()));
                                        ans /= static_cast<typename Tsor::value_type>(batch_size);
                                        return ans;
                                    }
                )( ex );
    }

    template <Expression Ex>
    auto constexpr reduce_mean( Ex const& ex ) noexcept
    {
        return mean_reduce( ex );
    }

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr minus( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        if constexpr (is_value_v<Rhs_Expression>)
        {
            return negative( plus( negative(lhs_ex), rhs_ex ) );
        }
        else
        {
            return plus( lhs_ex, negative(rhs_ex) );
        }
    }

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr operator - ( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return minus( lhs_ex, rhs_ex );
    }


    ///
    /// Returns the square of the input
    ///
    /// @param ex The input operator.
    /// @return An instance of a unary_operator that evaluate the squared value of the input operator.
    ///
    /// Example code:
    /// @code
    /// auto e = variable<tensor<float>>{ /*...*/ };
    /// auto square = square(e);
    /// @endcode
    ///
    template <Expression Ex>
    auto constexpr square( Ex const& ex ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& tsor ) noexcept
                                    {
                                        better_assert( !has_nan( tsor ), "forward propagation for operator square: tensor contains Nan!" );
                                        Tsor ans = tsor.deep_copy();
                                        std::for_each( ans.data(), ans.data() + ans.size(), []( auto & v ){ v *= v; } );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator square contains NaN!" );
                                        Tsor ans = input.deep_copy();
                                        ans *= grad;
                                        ans *= typename Tsor::value_type{2};
                                        return ans;
                                    }
                )( ex );
    }


    template <Expression Ex>
    auto constexpr abs( Ex const& ex ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& tsor ) noexcept
                                    {
                                        better_assert( !has_nan( tsor ), "forward propagation for operator abs: tensor contains Nan!" );
                                        Tsor ans = tsor.deep_copy();
                                        std::for_each( ans.data(), ans.data() + ans.size(), []( typename Tsor::value_type & v ){ v = std::abs(v); } );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator abs contains NaN!" );
                                        Tsor ans = grad;
                                        for ( auto idx : range( ans.size() ) )
                                            ans[idx] = (input[idx]>typename Tsor::value_type{0}) ? ans[idx] : -ans[idx];
                                        return ans;
                                    }
                )( ex );
    }//;

    template <Expression Ex>
    [[deprecated("GCC might die here. Use exponential instead.")]]
    auto constexpr exp( Ex const& ex ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& tsor ) noexcept
                                    {
                                        better_assert( !has_nan( tsor ), "forward propagation for operator exp: tensor contains Nan!" );
                                        Tsor ans = tsor.deep_copy();
                                        std::for_each( ans.data(), ans.data() + ans.size(), []( auto & v ){ v = std::exp(v); } );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator exp contains NaN!" );
                                        Tsor ans = grad;
                                        grad *= output;
                                        return ans;
                                    }
                )( ex );
    }

    template <typename Float> requires std::floating_point<Float>
    auto constexpr clip( Float lower, Float upper=std::numeric_limits<Float>::max() ) noexcept
    {
        return [lower, upper]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator( [lower, upper]<Tensor Tsor>( Tsor const& tsor ) noexcept
                                        {
                                            better_assert( !has_nan( tsor ), "forward propagation for operator clip: tensor contains Nan!" );
                                            Tsor ans = tsor.deep_copy();
                                            clip( ans, lower, upper );
                                            return ans;
                                        },
                                        [lower, upper]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                        {
                                            better_assert( !has_nan( grad ), "input gradient for operator clip contains NaN!" );
                                            const typename Tsor::value_type zero{0};
                                            Tsor ans = grad;
                                            for ( auto idx : range( input.size() ) )
                                                ans[idx] = (input[idx] < lower) ? zero :
                                                           (input[idx] > upper) ? zero :
                                                           ans[idx];
                                            return ans;
                                        }
                    )( ex );
        };
    }

    // include_batch_flag:
    //
    //  true: considering the batch size at the first dim
    //      - for an input of (1, 3, 4), expecting an incoming expression of shape like [BS, 12, 1 1]
    //      - expected output of shape [BS, 1, 3, 4]
    //  false: do not consider the batch size
    //      - for an input of (1, 3, 4), expecting an incoming expression of shape like [12, 1]
    //      - expected output of shape [1, 3, 4]
    auto inline reshape( std::vector<unsigned long> const& new_shape, bool include_batch_flag=true ) noexcept
    {
        return [new_shape, include_batch_flag]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [new_shape, include_batch_flag]<Tensor Tsor>( Tsor const& tsor ) noexcept
                {
                    unsigned long const new_size = std::accumulate( new_shape.begin(), new_shape.end(), 1UL, []( auto x, auto y ){ return x*y; } );
                    unsigned long const total_size = tsor.size();
                    unsigned long const batch_size = total_size / new_size;

                    better_assert( batch_size * new_size == total_size, "size mismatch for reshape operator, expect ",  batch_size*new_size, " but total input size is ", total_size, ", where batch_size is ", batch_size );

                    if ( !include_batch_flag )
                    {
                        better_assert( batch_size == 1, "expecting batch size of 1 while not including batch, but got ", batch_size );
                        Tsor ans{tsor};
                        ans.reshape( new_shape );
                        return ans;
                    }

                    std::vector<unsigned long> batched_new_shape;
                    {
                        batched_new_shape.resize( 1 + new_shape.size() );
                        batched_new_shape[0] = batch_size;
                        std::copy( new_shape.begin(), new_shape.end(), batched_new_shape.begin()+1 );
                    }

                    Tsor ans{ tsor };
                    ans.reshape( batched_new_shape );
                    return ans;
                },
                []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                {
                    Tsor ans{ grad };
                    ans.reshape( input.shape() );
                    return ans;
                }
            )( ex );
        };
    }

    template <Expression Ex>
    auto constexpr flatten( Ex const& ex ) noexcept
    {
        return make_unary_operator
        (
            []<Tensor Tsor>( Tsor const& tsor ) noexcept
            {
                better_assert( tsor.ndim() > 1, "Expecting dimension of incoming tensor to be greater than 1, but got ", tsor.ndim() );
                unsigned long const batch_size = *(tsor.shape().begin());
                unsigned long const rem = tsor.size() / batch_size;
                Tsor ans = tsor;
                return ans.reshape( {batch_size, rem} );
            },
            []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
            {
                Tsor ans = grad;
                return ans.reshape( input.shape() );
            }
        )( ex );
    }

    template <Expression Ex>
    auto constexpr identity( Ex const& ex ) noexcept
    {
        return make_unary_operator
        (
            []<Tensor Tsor>( Tsor const& tsor ) noexcept
            {
                return tsor;
            },
            []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
            {
                return grad;
            }
        )( ex );
    }

    template< Expression Ex >
    auto transpose( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator
        (
            [forward_cache]<Tensor Tsor>( Tsor const& tsor ) noexcept
            {
                better_assert( tsor.ndim() == 2, "Expecting 2D tensor, but got dimensions ", tsor.ndim() );

                typedef typename Tsor::value_type value_type;

                std::vector<unsigned long> const shape = tsor.shape();
                auto const[row, col] = std::make_tuple( shape[0], shape[1] );
                view_2d<value_type> v_in{ tsor.data(), row, col };

                Tsor& ans = context_cast<Tsor>( forward_cache );
                ans.resize( {col, row} );
                view_2d<value_type> v_out{ ans.data(), col, row };

                for ( auto r : range( row ) )
                    for ( auto c : range( col ) )
                        v_out[c][r] = v_in[r][c];

                return ans;
            },
            [backward_cache]<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
            {
                typedef typename Tsor::value_type value_type;

                std::vector<unsigned long> const shape = grad.shape();
                auto const[row, col] = std::make_tuple( shape[0], shape[1] );
                view_2d<value_type> v_in{ grad.data(), row, col };

                Tsor& back_ans = context_cast<Tsor>( backward_cache );
                back_ans.resize( {col, row} );

                view_2d<value_type> v_out{ back_ans.data(), col, row };

                for ( auto r : range( row ) )
                    for ( auto c : range( col ) )
                        v_out[c][r] = v_in[r][c];

                return back_ans;
            }
        )( ex );
    }

    auto inline img2col( unsigned long const row_kernel, unsigned long col_kernel=-1,
                         unsigned long const row_padding=0, unsigned long col_padding=0,
                         unsigned long const row_stride=1, unsigned long const col_stride=1,
                         unsigned long const row_dilation=1, unsigned long const col_dilation=1 ) noexcept
    {
        if ( col_kernel == (unsigned long)-1 ) col_kernel = row_kernel;

        std::shared_ptr<std::vector<std::uint32_t>> s_index_record = std::make_shared<std::vector<std::uint32_t>>(); // col_img[idx] = img[index_record[idx]]  -- (-1) for zero padding

        auto img2col_forward = [s_index_record]<Tensor Tsor>
        (
            Tsor const& input_img, Tsor& output_col_mat,
            unsigned long kernel_row, unsigned long kernel_col,
            unsigned long padding_row, unsigned long padding_col,
            unsigned long stride_row, unsigned long stride_col,
            unsigned long dilation_row, unsigned long dilation_col
        ) noexcept
        {
            typedef typename Tsor::value_type value_type;
            std::vector<std::uint32_t>& index_record = *s_index_record; //32 bit should be enough for memory address offeset

            std::vector<unsigned long> input_shape = input_img.shape();
            better_assert( input_shape.size() == 4, "Expecting a 4D tensor." );
            auto const [BS, R, C, CH] = std::make_tuple( input_shape[0], input_shape[1], input_shape[2], input_shape[3] );

            unsigned long const output_row = ( R + 2 * padding_row - ( dilation_row * (kernel_row - 1) + 1 ) ) / stride_row + 1;
            unsigned long const output_col = ( C + 2 * padding_col - ( dilation_col * (kernel_col - 1) + 1 ) ) / stride_col + 1;
            unsigned long const output_column_matrix_row = kernel_row * kernel_col * CH;
            unsigned long const output_column_matrix_col = BS * output_row * output_col;

            output_col_mat.resize( {output_column_matrix_row, output_column_matrix_col} );

            if ( index_record.size() != output_column_matrix_row * output_column_matrix_col ) // first-run?
            {
                index_record.resize( output_column_matrix_row * output_column_matrix_col );

                for ( auto bs : range( BS ) )
                {
                    std::int64_t const col_offset = bs * output_row * output_col * kernel_row * kernel_col * CH;
                    std::int64_t const im_offset = bs * R * C * CH;
                    for ( auto c : range( CH * kernel_row * kernel_col ) )
                    {
                        std::int64_t const w_offset = c % kernel_col;
                        std::int64_t const h_offset = ( c / kernel_col ) % kernel_row;
                        std::int64_t const c_im = c / ( kernel_col * kernel_row );

                        for ( auto h : range( output_row ) )
                        {
                            std::int64_t const im_row_idx = h * stride_row - padding_row + h_offset * dilation_row;
                            for ( auto w : range( output_col ) )
                            {
                                std::int64_t const im_col_idx = w * stride_col - padding_col + w_offset * dilation_col;
                                std::int64_t const im_idx = im_offset+( im_row_idx * C + im_col_idx ) * CH + c_im;
                                std::int64_t const col_idx = col_offset+( c * output_row + h ) * output_col + w;
                                index_record[col_idx] = static_cast<std::uint32_t>((im_row_idx<0 || im_row_idx>=static_cast<std::int64_t>(R) || im_col_idx<0 || im_col_idx>=static_cast<std::int64_t>(C)) ? 0xffffffff : im_idx);
                            }
                        }
                    }
                }
                // re-arrange [bs, new_R, new_C] --> [new_R, new_c*bs]
                {
                    std::vector<std::uint32_t> re_arranged_index;
                    re_arranged_index.resize( index_record.size() );

                    view_3d<std::uint32_t> re_arranged_mat{ re_arranged_index.data(), output_column_matrix_row, BS, output_row*output_col };
                    view_3d<std::uint32_t> index_record_mat{ index_record.data(), BS, output_column_matrix_row, output_row*output_col };

                    for ( auto bs : range( BS ) )
                        for ( auto r : range( output_column_matrix_row ) )
                            for ( auto c : range( output_row*output_col ) )
                                re_arranged_mat[r][bs][c] = index_record_mat[bs][r][c];
                    // overwrite index record
                    std::copy( re_arranged_index.begin(), re_arranged_index.end(), index_record.begin() );
                }
            }

            // fill-in
            for ( auto idx : range( output_col_mat.size() ) )
            {
                auto const index = index_record[idx];
                output_col_mat[idx] = (index == 0xffffffff) ? value_type{0} : input_img[index];
            }
        };

        auto img2col_backward = [s_index_record]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad, Tsor& ans ) noexcept
        {
            typedef typename Tsor::value_type value_type;
            ans.resize( input.shape() );
            std::fill( ans.begin(), ans.end(), value_type{0} );

            std::vector<std::uint32_t>& index_record = *s_index_record; //32 bit should be enough for memory address offeset
            for ( auto idx : range( grad.size() ) )
            {
                auto const index = index_record[idx];
                if ( index != 0xffffffff )
                    ans[index] += grad[idx];
            }
        };

        std::shared_ptr<std::any> output_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> back_grad_cache = std::make_shared<std::any>();

        return [row_kernel, col_kernel, row_padding, col_padding, row_stride, col_stride, row_dilation, col_dilation, img2col_forward, img2col_backward, output_cache, back_grad_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [=]<Tensor Tsor>( Tsor const & tsor ) noexcept
                {
                    Tsor& output = context_cast<Tsor>( output_cache );
                    img2col_forward( tsor, output, row_kernel, col_kernel, row_padding, col_padding, row_stride, col_stride, row_dilation, col_dilation );
                    return Tsor{output};
                },
                [=]<Tensor Tsor>( Tsor const& input, Tsor const& output, Tsor const& grad ) noexcept
                {
                    Tsor& back_grad = context_cast<Tsor>( back_grad_cache );
                    img2col_backward( input, output, grad, back_grad );
                    return Tsor{back_grad};
                }
            )( ex );
        };
    }

    auto inline conv2d
    (
        unsigned long row_input, unsigned long col_input,
        unsigned long const row_stride=1, unsigned long const col_stride=1,
        unsigned long const row_dilation=1, unsigned long const col_dilation=1,
        std::string const& padding="valid"
    ) noexcept
    {
        // lhs_ex is for one 4D tensor of [BS, R, C, CH]
        // rhs_ex is for NC 4D filter of [1, r, c, CH], thus the shape is [NC, r, c, CH]
        // the output tensor is of shape [BS, .., .., NC]
        //
        // Note: the rhs expression is fixed as a variable, as we need to extract the kernel shape from it
        //
        return [row_input, col_input, row_stride, col_stride, row_dilation, col_dilation, padding ]<Expression Ex, Variable Va>( Ex const& lhs_ex, Va const& rhs_ex ) noexcept
        {
            std::vector<unsigned long> const& shape = rhs_ex.shape();
            better_assert( shape.size() == 4 );
            auto const[new_channel, row_kernel, col_kernel, channel] = std::make_tuple( shape[0], shape[1], shape[2], shape[3] );
            //TODO: optimization in case of small kernels of (1, 1), (3, 3)
            unsigned long row_padding = 0;
            unsigned long col_padding = 0;
            if ( padding == "same" )
            {
                unsigned long const row_padding_total = (row_kernel + (row_kernel - 1) * (row_dilation - 1) - row_stride);
                better_assert( !(row_padding_total & 0x1), "Expecting total row padding to be even, but got ", row_padding_total, " With row input being ", row_input, " and row_stride ", row_stride );
                unsigned long const col_padding_total = (col_kernel + (col_kernel - 1) * (col_dilation - 1) - col_stride);
                better_assert( !(col_padding_total & 0x1), "Expecting total col padding to be even, but got ", col_padding_total );
                row_padding = ((row_kernel&1)+row_padding_total) >> 1;
                col_padding = ((col_kernel&1)+col_padding_total) >> 1;
            }

            unsigned long const row_output = ( row_input + 2 * row_padding - ( row_dilation * (row_kernel - 1) + 1 ) ) / row_stride + 1;
            unsigned long const col_output = ( col_input + 2 * row_padding - ( col_dilation * (col_kernel - 1) + 1 ) ) / col_stride + 1;

            auto lhs_ex_as_col = img2col(row_kernel, col_kernel, row_padding, col_padding, row_stride, col_stride, row_dilation, col_dilation)( lhs_ex ); // [BS, R, C, CH] ==> [r*c*CH, BS*new_row*new_col]

            auto rhs_ex_flatten = reshape({row_kernel*col_kernel*channel,})( rhs_ex ); // [NC, r, c, CH] ==> [NC, r*c*CH]

            auto flatten_output = rhs_ex_flatten * lhs_ex_as_col; // [NC, BS * new_row * new_col]

            auto tr_output = transpose( flatten_output ); // [BS*new_row*new_col, NC]

            auto ans = reshape({row_output, col_output, new_channel})( tr_output );

            return ans;
        };
    }

    template< typename T > requires std::floating_point<T>
    inline auto drop_out( T const factor ) noexcept
    {
        better_assert( factor < T{1}, "Expecting drop out rate less than 1, but got factor = ", factor );
        better_assert( factor > T{0}, "Expecting drop out rate greater than 0, but got factor = ", factor );

        std::shared_ptr<std::any> mask = std::make_shared<std::any>();
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [factor, mask, forward_cache, backward_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [factor, mask, forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                {
                    typedef typename Tsor::value_type value_type;

                    if ( learning_phase == 0 ) // defined in 'config.hpp'
                        return input;

                    std::any& mask_ = *mask;
                    // first run, initialize mask
                    if ( !mask_.has_value() )
                    {
                        Tsor const random_tensor = random<value_type>( input.shape() );
                        Tsor mask__{ input.shape() };
                        for ( auto idx : range( input.size() ) )
                            if ( random_tensor[ idx ] > factor )
                                mask__[ idx ] = 1;
                        mask_ = mask__; // initialize
                    }

                    Tsor& mask__ = std::any_cast<Tsor&>( mask_ );

                    Tsor& ans = context_cast<Tsor>( forward_cache );
                    ans.deep_copy( input );

                    for ( auto idx : range( input.size() ) )
                        ans[idx] *= mask__[idx] / (value_type{1} - factor);
                    return ans;
                },
                [mask, backward_cache]<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                {
                    if ( learning_phase == 0 ) // defined in 'config.hpp'
                        return grad;

                    Tsor& mask__ = std::any_cast<Tsor&>( *mask );

                    Tsor& ans = context_cast<Tsor>( backward_cache );
                    ans.deep_copy( grad );

                    for ( auto idx : range( grad.size() ) )
                        ans[idx] *= mask__[idx];
                    return ans;
                }
            )( ex );
        };
    }


    // comment: maybe using function 'reduce' to reduce the cod complexity? at a price of performance?
    inline auto max_pooling_2d( unsigned long stride ) noexcept
    {
        better_assert( stride > 1, "Expecting max_pooling_2d stride greater than 1, but got ", stride );

        std::shared_ptr<std::any> mask = std::make_shared<std::any>();
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [stride, mask, forward_cache, backward_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [stride, mask, forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept // [BS, R, C, CH] --> [BS, R/s, C/s, CH]
                {
                    typedef typename Tsor::value_type value_type;
                    better_assert( input.ndim() == 4, "Expecting a 4D tensor, but got ", input.ndim() );

                    Tsor& mask__ = context_cast<Tsor>( mask );
                    mask__.resize( input.shape() );


                    std::vector<unsigned long> shape = input.shape();
                    auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
                    Tsor input_ = input;
                    view_4d<value_type> ts{ input_.data(), batch_size, row, col, channel };
                    view_4d<value_type> tm{ mask__.data(), batch_size, row, col, channel };

                    Tsor& ans = context_cast<Tsor>( forward_cache );
                    ans.resize( {batch_size, row/stride, col/stride, channel} );

                    view_4d<value_type> t1{ ans.data(), batch_size, row/stride, col/stride, channel };

                    for ( auto bs : range(batch_size) )
                        for ( auto r : range(row/stride) ) // row for t1
                            for ( auto c : range(col/stride) ) // col for t1
                                for ( auto ch : range(channel) )
                                {
                                    unsigned long current_row_max = r * stride;
                                    unsigned long current_col_max = c * stride;
                                    for ( auto _r : range( (r*stride), ((r*stride)+stride) ) ) // row for ts
                                        for ( auto _c : range( (c*stride), ((c*stride)+stride) ) ) // col for ts
                                        {
                                            if ( ts[bs][_r][_c][ch] > ts[bs][current_row_max][current_col_max][ch] )
                                            {
                                                current_row_max = _r;
                                                current_col_max = _c;
                                            }
                                        }
                                    tm[bs][current_row_max][current_col_max][ch] = 1.0; //mark as max
                                    t1[bs][r][c][ch] = ts[bs][current_row_max][current_col_max][ch]; // update value
                                }
                    return ans;
                },
                [stride, mask, backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                {
                    typedef typename Tsor::value_type value_type;
                    std::vector<unsigned long> const& shape = input.shape();
                    auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);

                    Tsor& mask__ = std::any_cast<Tsor&>( *mask );
                    view_4d<value_type> tm{ mask__.data(), batch_size, row, col, channel };

                    Tsor& ans = context_cast<Tsor>( backward_cache );
                    ans.resize( input.shape() );

                    view_4d<value_type> ta{ ans.data(), batch_size, row, col, channel };

                    Tsor grad_ = grad;
                    view_4d<value_type> tg{ grad_.data(), batch_size, row/stride, col/stride, channel };

                    for ( auto bs : range( batch_size ) )
                        for ( auto r : range( row ) )
                            for ( auto c : range( col ) )
                                for ( auto ch : range( channel ) )
                                    if ( std::abs(tm[bs][r][c][ch] - 1.0) < 1.0e-5 )
                                        ta[bs][r][c][ch] = tg[bs][r/stride][c/stride][ch];
                    return ans;
                }
            )( ex );
        };
    }

    inline auto average_pooling_2d( unsigned long stride ) noexcept
    {
        better_assert( stride > 1, "Expecting average_pooling_2d stride greater than 1, but got ", stride );

        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [stride, forward_cache, backward_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [stride, forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept // [BS, R, C, CH] --> [BS, R/s, C/s, CH]
                {
                    typedef typename Tsor::value_type value_type;
                    better_assert( input.ndim() == 4, "Expecting a 4D tensor, but got ", input.ndim() );

                    std::vector<unsigned long> shape = input.shape();
                    auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
                    Tsor input_ = input;
                    view_4d<value_type> ts{ input_.data(), batch_size, row, col, channel };

                    Tsor& ans = context_cast<Tsor>( forward_cache );
                    ans.resize( {batch_size, row/stride, col/stride, channel} );
                    std::fill( ans.begin(), ans.end(), value_type{0} );

                    view_4d<value_type> t1{ ans.data(), batch_size, row/stride, col/stride, channel };

                    value_type const factor = value_type{1} / static_cast<value_type>(stride*stride);
                    for ( auto bs : range(batch_size) )
                        for ( auto r : range(row/stride) ) // row for t1
                            for ( auto c : range(col/stride) ) // col for t1
                                for ( auto ch : range(channel) )
                                    for ( auto _r : range( (r*stride), ((r*stride)+stride) ) ) // row for ts
                                        for ( auto _c : range( (c*stride), ((c*stride)+stride) ) ) // col for ts
                                            t1[bs][r][c][ch] += ts[bs][_r][_c][ch] * factor;
                    return ans;
                },
                [stride, backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                {
                    typedef typename Tsor::value_type value_type;
                    std::vector<unsigned long> const& shape = input.shape();
                    auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);

                    Tsor& ans = context_cast<Tsor>( backward_cache );
                    ans.resize( input.shape() );

                    view_4d<value_type> ta{ ans.data(), batch_size, row, col, channel };

                    Tsor grad_ = grad;
                    view_4d<value_type> tg{ grad_.data(), batch_size, row/stride, col/stride, channel };

                    value_type const factor = value_type{1} / static_cast<value_type>(stride*stride);
                    for ( auto bs : range( batch_size ) )
                        for ( auto r : range( row ) )
                            for ( auto c : range( col ) )
                                for ( auto ch : range( channel ) )
                                    ta[bs][r][c][ch] = factor * tg[bs][r/stride][c/stride][ch];
                    return ans;
                }
            )( ex );
        };
    }

    inline auto up_sampling_2d( unsigned long stride ) noexcept
    {
        better_assert( stride > 1, "Expecting up_sampling_pooling_2d stride greater than 1, but got ", stride );

        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [stride, forward_cache, backward_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [stride, forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept // [BS, R, C, CH] --> [BS, R/s, C/s, CH]
                {
                    typedef typename Tsor::value_type value_type;
                    better_assert( input.ndim() == 4, "Expecting a 4D tensor, but got ", input.ndim() );

                    std::vector<unsigned long> shape = input.shape();
                    auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
                    Tsor input_ = input;
                    view_4d<value_type> ts{ input_.data(), batch_size, row, col, channel };

                    Tsor& ans = context_cast<Tsor>( forward_cache );
                    ans.resize( {batch_size, row*stride, col*stride, channel} );
                    std::fill( ans.begin(), ans.end(), value_type{0} );

                    view_4d<value_type> t1{ ans.data(), batch_size, row*stride, col*stride, channel };

                    for ( auto bs : range(batch_size) )
                        for ( auto r : range(row) ) // row for ts
                            for ( auto c : range(col) ) // col for ts
                                for ( auto ch : range(channel) )
                                    for ( auto _r : range( (r*stride), ((r*stride)+stride) ) ) // row for t1
                                        for ( auto _c : range( (c*stride), ((c*stride)+stride) ) ) // col for t1
                                            t1[bs][_r][_c][ch] = ts[bs][r][c][ch];
                    return ans;
                },
                [stride, backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                {
                    typedef typename Tsor::value_type value_type;
                    std::vector<unsigned long> const& shape = input.shape();
                    auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);

                    Tsor& ans = context_cast<Tsor>( backward_cache );
                    ans.resize( input.shape() );
                    std::fill( ans.begin(), ans.end(), value_type{0} );

                    view_4d<value_type> ta{ ans.data(), batch_size, row, col, channel };

                    Tsor grad_ = grad;
                    view_4d<value_type> tg{ grad_.data(), batch_size, row*stride, col*stride, channel };

                    for ( auto bs : range( batch_size ) )
                        for ( auto r : range( row ) )
                            for ( auto c : range( col ) )
                                for ( auto ch : range( channel ) )
                                    for ( auto _r : range( (r*stride), ((r*stride)+stride) ) ) // row for tg
                                        for ( auto _c : range( (c*stride), ((c*stride)+stride) ) ) // col for tg
                                            ta[bs][r][c][ch] += tg[bs][_r][_c][ch];
                    return ans;
                }
            )( ex );
        };
    }


    template< typename T=double > requires std::floating_point<T>
    inline auto normalization_batch( T const momentum=0.98 ) noexcept
    {
        std::shared_ptr<std::any> global_average_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> global_variance_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> average_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> variance_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [=]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [=]<Tensor Tsor>( Tsor const& input ) noexcept
                {
                    better_assert( input.ndim() > 1, "normalization_batch requires input dimension at least 2, got ", input.ndim() );

                    typedef typename Tsor::value_type value_type;
                    //typedef typename Tsor::allocator allocator;

                    std::vector<unsigned long> const& shape = input.shape();
                    unsigned long const channels = *(shape.rbegin());
                    unsigned long const rest_dims = input.size() / channels;

                    view_2d<value_type> input_{ input.data(), rest_dims, channels };

                    // case of prediction phase, in this phase, the batch size could be 1, and it is not possible to calculate the variance
                    if ( learning_phase == 0 ) // defined in 'config.hpp'
                    {
                        // fix for the special case when prediction is executed before the training, typically in a GAN
                        Tsor& global_average_test = context_cast<Tsor>( global_average_cache );
                        if ( global_average_test.empty() )
                            return input;

                        // normal case. i.e., the global_average_cache and global_variance_cache are not empty
                        Tsor& global_average = context_extract<Tsor>( global_average_cache );
                        Tsor& global_variance = context_extract<Tsor>( global_variance_cache );

                        Tsor& ans = context_cast<Tsor>( forward_cache, zeros_like( input ) );
                        ans.resize( input.shape() ); // well, the batch sizes for training and for prediction are not necessarily same

                        view_2d<value_type> ans_{ ans.data(), rest_dims, channels };
                        {
                            for ( auto r : range( rest_dims ) )
                                for ( auto c : range( channels ) )
                                    ans_[r][c] = (input_[r][c] - global_average[c]) / std::sqrt( global_variance[c] + eps );
                        }
                        return ans;
                    }

                    //calculate average along the last channel
                    Tsor& average = context_cast<Tsor>( average_cache );
                    {
                        average.resize( {channels, } );
                        std::fill( average.begin(), average.end(), value_type{0} );

                        for ( auto idx : range( rest_dims ) )
                            for ( auto jdx : range( channels ) )
                                average[jdx] += input_[idx][jdx];

                        average /= static_cast<value_type>(rest_dims);
                    }

                    //calculate Variance along the last channel
                    Tsor& variance = context_cast<Tsor>( variance_cache );
                    {
                        variance.resize( {channels,} );
                        std::fill( variance.begin(), variance.end(), value_type{0} );
                        for ( auto idx : range( rest_dims ) )
                            for ( auto jdx : range( channels ) )
                                variance[jdx] += std::pow( input_[idx][jdx] - average[jdx], 2 );

                        variance /= static_cast<value_type>( rest_dims );
                    }


                    Tsor& ans = context_cast<Tsor>( forward_cache );
                    ans.resize( input.shape() ); // the batch sizes for training and for prediction are not necessarily same
                    view_2d<value_type> ans_{ ans.data(), rest_dims, channels };
                    {
                        for ( auto idx : range( rest_dims ) )
                            for ( auto jdx : range( channels ) )
                                ans_[idx][jdx] = ( input_[idx][jdx] - average[jdx] ) / std::sqrt( variance[jdx] + eps );
                    }

                    // update global average and global variance
                    {
                        Tsor& global_average = context_cast<Tsor>( global_average_cache, zeros_like( average ) );
                        // Note: No obvious different is observed between initializing global_variance to zeros and to ones with MNIST example:
                        //       initializing global_variance to zeros, after 10 epochs mnist gives an error of 0.026
                        //       initializing global_variance to ones, after 10 epochs mnist gives an error of 0.028
                        Tsor& global_variance = context_cast<Tsor>( global_variance_cache, zeros_like( variance ) );
                        //Tsor& global_variance = context_cast<Tsor>( global_variance_cache, ones_like( variance ) );
                        for ( auto idx : range( global_average.size() ) )
                        {
                            global_average[idx] = global_average[idx] * momentum + average[idx] * ( 1.0 - momentum );
                            global_variance[idx] = global_variance[idx] * momentum + variance[idx] * ( 1.0 - momentum );
                        }
                    }

                    return ans;
                },

                [=]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                {
                    typedef typename Tsor::value_type value_type;
                    Tsor& variance = context_extract<Tsor>( variance_cache );

                    std::vector<unsigned long> const& shape = input.shape();
                    unsigned long const channels = *(shape.rbegin());
                    unsigned long const rest_dims = input.size() / channels;

                    Tsor& ans = context_cast<Tsor>( backward_cache, zeros_like( input ) );
                    view_2d<value_type> ans_{ans.data(), rest_dims, channels };
                    view_2d<value_type> grad_{grad.data(), rest_dims, channels };
                    for ( auto r : range( rest_dims ) )
                        for ( auto c : range( channels ) )
                            ans_[r][c] = grad_[r][c] / std::sqrt( variance[c] + eps );
                    return ans;
                }
            )( ex );
        };
    }



    template< typename T > requires std::floating_point<T>
    inline auto batch_normalization( T const momentum=0.98 ) noexcept
    {
        return [=]<Expression Ex, Variable Va>( Ex const& ex, Va const& gamma, Va const& beta ) noexcept
        {
            return elementwise_product( normalization_batch(momentum)(ex), gamma ) + beta; // multiply and sum along the batch: normalization is of shape [BS, R, C, CH], gamma/beta are of shape [R, C, CH]
        };
    }






    //
    //  example:
    //
    //      variable<tensor<float>> a {... };
    //      variable<tensor<float>> b {... };
    //      auto cab = concatenate( a, b )();
    //
    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr concatenate( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return [&]( unsigned long axe = -1 ) noexcept
        {
            return make_binary_operator
            (
                [axe]<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
                {
                    return concatenate( lhs_tensor, rhs_tensor, axe );
                },
                [axe]<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const grad ) noexcept
                {
                    typedef typename Tsor::value_type value_type;

                    Tsor l_ans{ lhs_input.shape() };
                    Tsor r_ans{ rhs_input.shape() };
                    better_assert(  l_ans.size() + r_ans.size() == grad.size(), "size mismatch: lhs size is ", l_ans.size(), " rhs size is ", r_ans.size(), " and grad size is ", grad.size(),
                                    " with lhs dim is ", l_ans.ndim(), "  and rhs dim is ", r_ans.ndim() );

                    // 2D view of grad
                    unsigned long const ax = (axe == (unsigned long)(-1)) ? grad.ndim()-1 : axe;
                    unsigned long const g_col = std::accumulate( grad.shape().begin()+ax, grad.shape().end(), 1UL, []( unsigned long x, unsigned long y ){ return x*y; } );
                    unsigned long const g_row = grad.size() / g_col;
                    view_2d<value_type> v_g{ grad.data(), g_row, g_col };

                    // 2D view of l_ans
                    unsigned long const lhs_row = g_row;
                    unsigned long const lhs_col = lhs_input.size() / lhs_row;
                    view_2d<value_type> v_l{ l_ans.data(), lhs_row, lhs_col };

                    // 2D view of r_ans
                    unsigned long const rhs_row = g_row;
                    unsigned long const rhs_col = rhs_input.size() / rhs_row;
                    view_2d<value_type> v_r{ r_ans.data(), rhs_row, rhs_col };

                    better_assert( g_col == lhs_col + rhs_col, "last dimension not agree" );

                    for ( unsigned long idx = 0; idx != g_row; ++idx )
                    {
                        std::copy( v_g[idx], v_g[idx]+lhs_col, v_l[idx] );          // fill idx-th row of 'v_l'
                        std::copy( v_g[idx]+lhs_col, v_g[idx]+g_col, v_r[idx] );    // fill idx-th row of 'v_r'
                    }

                    return std::make_tuple( l_ans, r_ans );
                },
                "Concatenate"
            )( lhs_ex, rhs_ex );
        };
    }

    // just to keep this interface agrees with Keras
    inline auto concatenate( unsigned long axe = -1 )
    {

        return [=]< Expression Lhs_Expression, Expression Rhs_Expression >( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
        {
            return concatenate( lhs_ex, rhs_ex )( axe );
        };
    }

    // alias of 'concatenate'
    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr concat( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return concatenate( lhs_ex, rhs_ex );
    }

    // alias of 'concatenate'
    inline auto concat( unsigned long axe = -1 )
    {
        return concatenate( axe );
    }

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr maximum( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> mask_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache_lhs = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache_rhs = std::make_shared<std::any>();
        return make_binary_operator
        (
            [=]<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
            {
                better_assert( lhs_tensor.shape() == rhs_tensor.shape(), "tensor shape mismatch." );

                Tsor& ans = context_cast<Tsor>( forward_cache );
                ans.resize( lhs_tensor.shape() );
                Tsor& mask = context_cast<Tsor>( mask_cache ); // 1 if lhs element is larger, 0 if rhs element is larger
                mask.resize( lhs_tensor.shape() );

                for_each( lhs_tensor.begin(), lhs_tensor.end(), rhs_tensor.begin(), ans.begin(), mask.begin(), []( auto const l, auto const r, auto& a, auto& m ) { m = l > r ? 1.0 : 0.0; a = l > r ? l : r; } );

                return ans;
            },
            [=]<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const& grad ) noexcept
            {
                Tsor& mask = context_cast<Tsor>( mask_cache ); // 1 if lhs element is larger, 0 if rhs element is larger

                Tsor& l_ans = context_cast<Tsor>( backward_cache_lhs );
                l_ans.resize( lhs_input.shape() );
                Tsor& r_ans = context_cast<Tsor>( backward_cache_rhs );
                r_ans.resize( rhs_input.shape() );

                for_each( grad.begin(), grad.end(), mask.begin(), l_ans.begin(), r_ans.begin(), []( auto const g, auto const m, auto& l, auto& r ) { if ( m > 0.5 ) { l = g; r = 0.0; } else { l = 0.0; r = g; } } );

                return std::make_tuple( l_ans, r_ans );
            },
            "Maximum"
        )( lhs_ex, rhs_ex );
    }

    ///
    /// `random_normal_like` produces random tensor from a normal distribution
    /// @param mean Mean of the normal distribution, a scalar.
    /// @param stddev Standard deviation of the normal distribution, a scalar.
    /// @return An unary operator that takes an unary operator, and producing output tensor from a normal distribution. The shape of the output tensor has the same shape corresponding to the input unary operator.
    ///
    /// Example Code
    /// @code
    /// auto va = variable{ ones<float>({3, 3, 3}) };
    /// auto v_rand = random_normal_like( 1.0, 4.0 )( va ); // this expression will produces a tensor of shape (3, 3, 3) from a normal distribution with parameters (1.0, 4.0)
    /// @endcode
    ///
    template< typename T=float > requires std::floating_point<T>
    inline auto random_normal_like( T mean = 0.0, T stddev = 1.0 ) noexcept
    {
        return [=]<Expression Ex>(Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [=]<Tensor Tsor>( Tsor const& tsor ) noexcept
                {
                    //debug_log( "Trying to generate random variables from a normal distribution of mean ", mean, " and stddev ", stddev );
                    return randn_like( tsor, mean, stddev );
                },
                []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                {
                    return zeros_like( grad );
                },
                "RandomNormalLike"
            )(ex);
        };
    }

    ///
    /// Returns the truth value of (lhs == rhs) element-wise. [+1 for true, 0 for false]
    ///
    /// @param lhs_ex The first operator.
    /// @param rhs_ex The second operator.
    /// @return An instance of a binary operator that evaluate the element-wise equality of two input operators.
    ///
    /// Example code:
    /// @code
    /// auto l = variable<tensor<float>>{ /*...*/ };
    /// auto r = place_holder<tensor<float>>{};
    /// auto eq = equal(l, r);
    /// @endcode
    ///
    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr equal( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_binary_operator
        (
            [=]<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
            {
                typedef typename Tsor::value_type value_type;
                better_assert( lhs_tensor.shape() == rhs_tensor.shape(), "equal: tensor shape mismatch." );

                Tsor& ans = context_cast<Tsor>( forward_cache );
                ans.resize( lhs_tensor.shape() );
                for_each( lhs_tensor.begin(), lhs_tensor.end(), rhs_tensor.begin(), ans.begin(), []( auto l, auto r, auto& v ){ v = (std::abs(l-r) > eps) ? value_type{0} : value_type{1}; } );
                return ans;
            },
            [=]<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const& grad ) noexcept
            {
                // note: tensorflow has no gradient for operator Equal
                typedef typename Tsor::value_type value_type;
                Tsor& ans = context_cast<Tsor>( backward_cache );
                ans.resize( lhs_input.shape() );
                for_each( lhs_input.begin(), lhs_input.end(), rhs_input.begin(), grad.begin(), ans.begin(), []( auto l, auto r, auto g, auto& v ){ v = (std::abs(l-r) > eps) ? value_type{0} : value_type{g}; } );
                return std::make_tuple( ans, ans );
            },
            "Equal"
        )( lhs_ex, rhs_ex );
    }

    ///
    /// Returns the sign. [1 for positive, 0 for 0 and -1 for negative]
    ///
    /// @param ex The input operator.
    /// @return An instance of a unary_operator that evaluate the sign of the input operator.
    ///
    /// Example code:
    /// @code
    /// auto e = variable<tensor<float>>{ /*...*/ };
    /// auto si = sign(e);
    /// @endcode
    ///
    template <Expression Ex>
    auto constexpr sign( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator
        (
            [=]<Tensor Tsor>( Tsor const& input ) noexcept
            {
                typedef typename Tsor::value_type value_type;
                Tsor& ans = context_cast<Tsor>( forward_cache );
                ans.resize( input.shape() );
                for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ){ v = (value_type{0} < x) - (x < value_type{0}); } );
                return ans;
            },
            [=]<Tensor Tsor>( Tsor const&input, Tsor const&, Tsor const& grad ) noexcept
            {
                typedef typename Tsor::value_type value_type;
                Tsor& ans = context_cast<Tsor>( backward_cache );
                ans.resize( input.shape() );
                std::fill( ans.begin(), ans.end(), value_type{0} ); //TF gives zeros, we follow TF here
                return ans;
            },
            "Sign"
        )( ex );
    };



#if 0
    // TODO: fix this layer
    inline auto normalization_layer() noexcept
    {
        std::shared_ptr<std::any> average_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> variance_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [=]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [=]<Tensor Tsor>( Tsor const& input ) noexcept
                {
                    better_assert( input.ndim() > 1, "normalization_layer requires input dimension at least 2, got ", input.ndim() );

                    typedef typename Tsor::value_type value_type;
                    //typedef typename Tsor::allocator allocator;

                    std::vector<unsigned long> const& shape = input.shape();
                    unsigned long const channels = *(shape.rbegin());
                    unsigned long const rest_dims = input.size() / channels;

                    view_2d<value_type> input_{ input.data(), rest_dims, channels };

                    //calculate average along the last channel
                    Tsor& average = context_cast<Tsor>( average_cache );
                    {
                        average.resize( {channels, } );
                        std::fill( average.begin(), average.end(), value_type{0} );

                        for ( auto idx : range( rest_dims ) )
                            for ( auto jdx : range( channels ) )
                                average[jdx] += input_[idx][jdx];

                        average /= static_cast<value_type>(rest_dims);
                    }

                    //calculate Variance along the last channel
                    Tsor& variance = context_cast<Tsor>( variance_cache );
                    {
                        variance.resize( {channels,} );
                        std::fill( variance.begin(), variance.end(), value_type{0} );
                        for ( auto idx : range( rest_dims ) )
                            for ( auto jdx : range( channels ) )
                                variance[jdx] += std::pow( input_[idx][jdx] - average[jdx], 2 );

                        variance /= static_cast<value_type>( rest_dims );
                    }

                    Tsor& ans = context_cast<Tsor>( forward_cache );
                    ans.resize( input.shape() ); // the batch sizes for training and for prediction are not necessarily same
                    view_2d<value_type> ans_{ ans.data(), rest_dims, channels };
                    {
                        for ( auto idx : range( rest_dims ) )
                            for ( auto jdx : range( channels ) )
                                ans_[idx][jdx] = ( input_[idx][jdx] - average[jdx] ) / std::sqrt( variance[jdx] + eps );
                    }

                    return ans;
                },

                [=]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                {
                    typedef typename Tsor::value_type value_type;
                    Tsor& variance = context_extract<Tsor>( variance_cache );

                    std::vector<unsigned long> const& shape = input.shape();
                    unsigned long const channels = *(shape.rbegin());
                    unsigned long const rest_dims = input.size() / channels;

                    Tsor& ans = context_cast<Tsor>( backward_cache, zeros_like( input ) );
                    view_2d<value_type> ans_{ans.data(), rest_dims, channels };
                    view_2d<value_type> grad_{grad.data(), rest_dims, channels };
                    for ( auto r : range( rest_dims ) )
                        for ( auto c : range( channels ) )
                            ans_[r][c] = grad_[r][c] / std::sqrt( variance[c] + eps );
                    return ans;
                },
                "LayerNormalization"
            )( ex );
        };
    }

    inline auto layer_normalization() noexcept
    {
        return [=]<Expression Ex, Variable Va>( Ex const& ex, Va const& gamma, Va const& beta ) noexcept
        {
            return elementwise_product( normalization_layer()(ex), gamma ) + beta; // multiply and sum along the batch: normalization is of shape [BS, R, C, CH], gamma/beta are of shape [R, C, CH]
        };
    }
#endif

}//namespace ceras

#endif//IPKVWSJOCMGGVRASCBLPYHFBCHRIVEXYBOMMDAKFAUDFYVYOOOISLRXJNUJKPJEVMLDPRDSNM

