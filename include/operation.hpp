#ifndef IPKVWSJOCMGGVRASCBLPYHFBCHRIVEXYBOMMDAKFAUDFYVYOOOISLRXJNUJKPJEVMLDPRDSNM
#define IPKVWSJOCMGGVRASCBLPYHFBCHRIVEXYBOMMDAKFAUDFYVYOOOISLRXJNUJKPJEVMLDPRDSNM

#include "./includes.hpp"
#include "./place_holder.hpp"
#include "./variable.hpp"
#include "./utils/range.hpp"
#include "./utils/debug.hpp"

namespace ceras
{
    // an operator is composed of
    // 1. a left operator, a right operator and a lambda function, OR
    // 2. an operator and a lambda function

    // TODO: rewrite using overload idiom
    struct operator_type_wrapper
    {
        //default type with default shallow copy, as
        template< typename T >
        T operator()( T const& t ) const noexcept { return t; };

        template< typename T >
        std::reference_wrapper<T> operator()( T & t ) const noexcept { return std::ref(t); };

        //in case of a place holder, copy its reference, as place holder is not yet binded to a tensor already
        template< typename T, typename A >
        std::reference_wrapper<place_holder<T,A> const> operator()( place_holder<T, A> const& ph ) const noexcept
        {
            return std::cref( ph );
        }
    };

    struct forward_wrapper
    {
        template< typename T >
        auto operator() ( T& t ) const noexcept { return t.forward(); }

        template< typename T, typename A >
        auto operator() ( place_holder<T, A> const& ph ) const noexcept { return ph.forward(); }

        template< typename T >
        auto operator() ( std::reference_wrapper<T const> t ) const noexcept { return t.get().forward(); };

        template< typename T >
        auto operator() ( std::reference_wrapper<T> t ) const noexcept { return t.get().forward(); };
    };

    struct backward_wrapper
    {
        template< typename Op > // here Operation can also be variable
        auto operator() ( Op& op ) const noexcept
        {
            return [&op]<Tensor Tsor>( Tsor const& grad )
            {
                op.backward(grad);
            };
        }

        template< typename T, typename A > // here T for place holder
        auto operator() ( std::reference_wrapper<place_holder<T,A> const> ) const noexcept { return [](auto){}; }; //for place_holder

        template< typename T, typename A > // here T for place holder
        auto operator() ( place_holder<T,A> ) const noexcept { return [](auto){}; }; //for place_holder

        template< typename Op > // Operation and also variable
        auto operator() ( std::reference_wrapper<Op> op ) noexcept
        {
            return [op]<Tensor Tsor>(Tsor const& grad)
            {
                op.get().backward(grad);
            };
        }
    };

    // TODO:
    // 1. in forward propagation, output of one layer is not pasted to the next layer, try to optimize here

    template< typename Operator, typename Forward_Action, typename Backward_Action >
    struct unary_operator
    {
        decltype( operator_type_wrapper{}( std::declval<Operator>() ) ) op_;
        Forward_Action forward_action_;
        Backward_Action backward_action_;

        typedef decltype( std::declval<Forward_Action>()( std::declval<decltype( forward_wrapper{}(op_))>() ) ) tensor_type;
        tensor_type input_data_;
        tensor_type output_data_;

        std::vector<tensor_type> context_;

        unary_operator( Operator const& op, Forward_Action const& forward_action, Backward_Action const& backward_action ) noexcept :
            op_{operator_type_wrapper{}(op)}, forward_action_{ forward_action }, backward_action_{ backward_action } {}

        // update output_data, reset gradient
        auto forward()// const
        {
            input_data_ = forward_wrapper{}( op_ );
            if constexpr( std::is_invocable<Forward_Action, tensor_type const&>::value )
            {
                output_data_ = forward_action_( input_data_ );
            }
            else if constexpr( std::is_invocable<Forward_Action, tensor_type const&, std::vector<tensor_type>&>::value )
            {
                output_data_ = forward_action_( input_data_, context_ );
            }
            else
            {
                better_assert( false, "Should not be here!" );
            }

            return output_data_;
        }

        // update gradient
        template< typename T, typename A >
        void backward( tensor<T,A> const& grad )
        {
            if constexpr( std::is_invocable<Backward_Action, tensor_type const&, tensor_type const&, tensor_type const&>::value )
            {
                auto const& current_gradient = backward_action_( input_data_, output_data_, grad );
                backward_wrapper{}( op_ )( current_gradient );
            }
            else if constexpr( std::is_invocable<Backward_Action, tensor_type const&, tensor_type const&, tensor_type const&, std::vector<tensor_type>&>::value )
            {
                auto const& current_gradient = backward_action_( input_data_, output_data_, grad, context_ );
                backward_wrapper{}( op_ )( current_gradient );
            }
            else
            {
                better_assert( false, "Should not be here!" );
            }
        }
    };

    static auto constexpr make_unary_operator = []( auto const& unary_forward_action, auto const& unary_backward_action ) noexcept
    {
        return [&unary_forward_action, &unary_backward_action]( auto const& op ) noexcept
        {
            return unary_operator{ op, unary_forward_action, unary_backward_action };
        };
    };

    template< typename Lhs_Operator, typename Rhs_Operator, typename Forward_Action, typename Backward_Action >
    struct binary_operator
    {
        decltype( operator_type_wrapper{}( std::declval<Lhs_Operator>() ) ) lhs_op_;
        decltype( operator_type_wrapper{}( std::declval<Rhs_Operator>() ) ) rhs_op_;
        Forward_Action const& forward_action_;
        Backward_Action const& backward_action_; // backward action for binary operator produces a tuple of two tensors

        typedef decltype( std::declval<Forward_Action>()( std::declval<decltype( forward_wrapper{}(lhs_op_))>(), std::declval<decltype( forward_wrapper{}(rhs_op_))>() ) ) tensor_type;
        tensor_type lhs_input_data_;
        tensor_type rhs_input_data_;
        tensor_type output_data_;

        std::vector<tensor_type> context_;

        binary_operator( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op, Forward_Action const& forward_action, Backward_Action const& backward_action ) noexcept :
            lhs_op_{operator_type_wrapper{}(lhs_op)}, rhs_op_{operator_type_wrapper{}(rhs_op)}, forward_action_{ forward_action }, backward_action_{ backward_action } {}

        auto forward()// const
        {
            lhs_input_data_ = forward_wrapper{}( lhs_op_ );
            rhs_input_data_ = forward_wrapper{}( rhs_op_ );

            if constexpr( std::is_invocable<Forward_Action, tensor_type const&, tensor_type const&>::value )
            {
                output_data_ = forward_action_( lhs_input_data_, rhs_input_data_ );
            }
            else if constexpr( std::is_invocable<Forward_Action, tensor_type const&, tensor_type const&, std::vector<tensor_type>&>::value )
            {
                output_data_ = forward_action_( lhs_input_data_, rhs_input_data_, context_ );
            }
            else
            {
                better_assert( false, "Should not be here!" );
            }

            //output_data_ = forward_action_( lhs_input_data_, rhs_input_data_ );
            return output_data_;
        }

        template< typename T, typename A >
        void backward( tensor<T,A> const& grad )
        {
            if constexpr( std::is_invocable<Backward_Action, tensor_type const&, tensor_type const&, tensor_type const&, tensor_type const&>::value )
            {
                auto const& [current_gradient_lhs, current_gradient_rhs] = backward_action_( lhs_input_data_, rhs_input_data_, output_data_, grad );
                backward_wrapper{}( lhs_op_ )( current_gradient_lhs );
                backward_wrapper{}( rhs_op_ )( current_gradient_rhs );
            }
            else if constexpr( std::is_invocable<Backward_Action, tensor_type const&, tensor_type const&, tensor_type const&, tensor_type const&, std::vector<tensor_type>&>::value)
            {
                auto const& [current_gradient_lhs, current_gradient_rhs] = backward_action_( lhs_input_data_, rhs_input_data_, output_data_, grad, context_ );
                backward_wrapper{}( lhs_op_ )( current_gradient_lhs );
                backward_wrapper{}( rhs_op_ )( current_gradient_rhs );
            }
            else
            {
                better_assert( false, "Should not be here!" );
            }

            //auto const& [current_gradient_lhs, current_gradient_rhs] = backward_action_( lhs_input_data_, rhs_input_data_, output_data_, grad );
            //backward_wrapper{}( lhs_op_ )( current_gradient_lhs );
            //backward_wrapper{}( rhs_op_ )( current_gradient_rhs );
        }
    };

    static auto constexpr make_binary_operator = []( auto const& binary_forward_action, auto const& binary_backward_action ) noexcept
    {
        return [&binary_forward_action, &binary_backward_action]( auto const& lhs_op, auto const& rhs_op ) noexcept
        {
            return binary_operator{ lhs_op, rhs_op, binary_forward_action, binary_backward_action };
        };
    };

    template< typename T >
    struct is_operator : std::false_type {};

    template< typename Lhs_Operator, typename Rhs_Operator, typename Forward_Action, typename Backward_Action >
    struct is_operator< binary_operator<Lhs_Operator, Rhs_Operator, Forward_Action, Backward_Action> > : std::true_type {};

    template< typename Operator, typename Forward_Action, typename Backward_Action >
    struct is_operator< unary_operator<Operator, Forward_Action, Backward_Action> > : std::true_type {};

    template< class T >
    inline constexpr bool is_operator_v = is_operator<T>::value;

    template< typename T >
    concept Operator = is_operator_v<T>;


    template< typename T >
    concept Expression = Operator<T> || Variable<T> || Place_Holder<T>;

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr plus( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return make_binary_operator( []<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
                                     {
                                        better_assert( !has_nan( lhs_tensor ), "forward propagation for operator plus: lhs_tensor contains Nan!" );
                                        better_assert( !has_nan( rhs_tensor ), "forward propagation for operator plus: rhs_tensor contains Nan!" );
                                        return add( lhs_tensor, rhs_tensor );
                                     },
                                     []<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const grad ) noexcept
                                     {
                                        better_assert( !has_nan( grad ), "backprop: upcoming gradient for operator + contains NaN!" );

                                        auto const& grad_fun = [&grad]( auto const& input )
                                        {
                                            Tsor ans = grad.deep_copy();
                                            while( input.ndim() < ans.ndim() )
                                            {
                                                ans = sum( ans, 0 );
                                            }
                                            auto const& shape = input.shape();
                                            for ( auto axis : range( input.ndim() ) )
                                            {
                                                if ( shape[axis] == 1 )
                                                {
                                                    ans = sum( ans, axis, true );
                                                }
                                            }
                                            return ans;
                                        };
                                        return std::make_tuple( grad_fun( lhs_input), grad_fun( rhs_input ) );
                                     }
                )( lhs_ex, rhs_ex );
    }

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr operator + ( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return plus( lhs_ex, rhs_ex );
    }

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr operator * ( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return make_binary_operator( []<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
                                     {
                                        better_assert( !has_nan( lhs_tensor ), "forward propagation for operator *: lhs_tensor contains Nan!" );
                                        better_assert( !has_nan( rhs_tensor ), "forward propagation for operator *: rhs_tensor contains Nan!" );
                                        return multiply( lhs_tensor, rhs_tensor );
                                     },
                                     []<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const grad ) noexcept
                                     {
                                        better_assert( !has_nan( grad ), "backprop: input gradient for operator * contains NaN!" );
                                        // left branch <-- grad * rhs^T
                                        auto const& g_shape = grad.shape();
                                        auto const[m, n] = std::make_tuple( g_shape[0], g_shape[1] ); // 4, 1
                                        auto const k = *(lhs_input.shape().rbegin()); // 13
                                        Tsor lhs_grad{ lhs_input.shape() };
                                        gemm( grad.data(), false, rhs_input.data(), true, m, n, k, lhs_grad.data() );

                                        better_assert( !has_nan( lhs_grad ), "backprop: input gradient for operator * -- lhs result contains NaN!" );

                                        // right branch <-- lhs^T * grad
                                        Tsor rhs_grad{ rhs_input.shape() };
                                        gemm( lhs_input.data(), true, grad.data(), false, k, m, n, rhs_grad.data() );
                                        better_assert( !has_nan( rhs_grad ), "backprop: input gradient for operator * -- rhs result contains NaN!" );


                                        return std::make_tuple( lhs_grad, rhs_grad );
                                     }
                )( lhs_ex, rhs_ex );
    }


    template <Expression Ex>
    auto constexpr log( Ex const& ex ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        better_assert( !has_nan( input ), "forward propagation for operator log: input contains Nan!" );
                                        auto ans = input.deep_copy();
                                        ans.map( [](auto & x){ x = std::log(x); } );
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
    auto constexpr elementwise_multiply( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return make_binary_operator( []<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
                                     {
                                        better_assert( !has_nan( lhs_tensor ), "forward propagation for operator elementwise_multiply: lhs_tensor contains Nan!" );
                                        better_assert( !has_nan( rhs_tensor ), "forward propagation for operator elementwise_multiply: rhs_tensor contains Nan!" );
                                        return elementwise_product( lhs_tensor, rhs_tensor );
                                     },
                                     []<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const grad ) noexcept
                                     {
                                        better_assert( !has_nan( grad ), "input gradient for operator elementwise_multiply contains NaN!" );
                                        return std::make_tuple( elementwise_product(grad, rhs_input), elementwise_product(grad, lhs_input) );
                                     }
                )( lhs_ex, rhs_ex );
    };

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr hadamard_product( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return elementwise_multiply( lhs_ex, rhs_ex );
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
                                        std::size_t const batch_size = (input.shape().size() == 1) ? 1 : (*(input.shape().begin()));
                                        ans /= static_cast<typename Tsor::value_type>(batch_size);
                                        return ans;
                                    }
                )( ex );
    }

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr minus( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return plus( lhs_ex, negative(rhs_ex) );
    }

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
    auto constexpr clip( Float lower, Float upper ) noexcept
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

    auto constexpr reshape( std::vector<std::size_t> const& new_shape ) noexcept
    {
        return [&new_shape]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator( [&new_shape]<Tensor Tsor>( Tsor const& tsor ) noexcept
                                        {
                                            std::vector<std::size_t> const& old_shape = tsor.shape();
                                            std::size_t const batch_size = old_shape[0];
                                            {
                                                std::size_t const new_size_per_batch = std::accumulate( new_shape.begin(), new_shape.end(), 1UL, []( auto x, auto y ){ return x*y; } );
                                                better_assert( batch_size * new_size_per_batch == tsor.size(), "size mismatch for reshape operator, got ",  batch_size*new_size_per_batch, " but input is ", tsor.size() );
                                            }

                                            std::vector<std::size_t> batched_new_shape;
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
                                            return grad.reshape( input.shape() );
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
                std::size_t const batch_size = *(tsor.shape().begin());
                std::size_t const dim = tsor.size() / batch_size;
                return tsor.reshape( {batch_size, dim} );
            },
            []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
            {
                return grad.reshape( input.shape() );
            }
        )( ex );
    }


    auto constexpr img2col( std::size_t const row_stride, std::size_t const col_stride, std::size_t const kernel_row, std::size_t const kernel_col, std::string const& padding ) noexcept
    {
        auto const& pixel_at = []<Tensor Tsor>( Tsor const& tsor, std::size_t const bs, std::size_t const r, std::size_t const c, std::size_t const ch, std::size_t pading_row, std::size_t pading_col ) noexcept
        {
            //
            //tsor is a 4D tensor of shape [BS, R, C, CH]
            //this function is to retrieve the pixel at (bs, r, c, ch) with padding (padding_row, padding_col)
            //

            better_assert( tsor.ndim() == 4, "Expecting a 4D tensor, but actual dimension is ", tsor.ndim() );
            typedef typename Tsor::value_type value_type;

            if ( r < padding_row ) return value_type{0};
            if ( c < padding_col ) return value_type{0};

            std::size_t const row = r - padding_row;
            std::size_t const col = c - padding_col;

            std::vector<std::size_t> const& shape = tsor.shape();
            auto const[BS, R, C, CH] = std::make_tuple( shape[0], shape[1], shape[2], shape[3] );

            if ( row >= R ) return value_type{0};
            if ( col >= C ) return value_type{0};

            std::size_t const offset = ch + CH * (c + C* (r + bs * R));
            return *(tsor.begin() + offset)
        };

        // TODO: move this to GPU
        auto const& img2col = [&pixel_at, row_stride, col_stride, kernel_row, kernel_col]<Tensor Tsor>
        ( Tsor const& input, Tsor& output, std::size_t const padding_row, std::size_t const padding_col ) noexcept
        {
            better_assert( input.ndim() == 4, "Expecting a 4D tensor, but actual dimension is ", input.ndim() );

            std::vector<std::size_t> const& shape = tsor.shape();
            auto const[BS, R, C, CH] = std::make_tuple( shape[0], shape[1], shape[2], shape[3] );

            std::size_t const new_row = static_cast<std::size_t>( (R + padding_row + padding_row - kernel_row) / row_stride ) + 1;
            std::size_t const new_col = static_cast<std::size_t>( (C + padding_col + padding_col - kernel_col) / col_stride ) + 1;

            output.reshape( {BS*new_row*new_col, kernel_row*kernel_col*CH} );

            // dat = input[bs, r*stride_row:stride_row:(r+kernel_row)*stride_row, c*stride_col:stride_col:(c+kernel_col)*stride_col, : ]
            std::size_t const stride_kc = CH;
            std::size_t const stride_kr = kernel_col * stride_kc;
            std::size_t const stride_c = kernel_row * stride_kr;
            std::size_t const stride_r = new_col * stride_c;
            std::size_t const stride_bs = new_row * stride_r;
            for ( auto bs : range( BS ) )
            {
                std::size_t const offset_bs = bs * stride_bs;
                for ( auto r : range( new_row ) )
                {
                    std::size_t const offset_r = offset_bs + r * stride_r;
                    for ( auto c : range( new_col ) )
                    {
                        std::size_t const offset_c =  offset_r + c * stride_c;
                        for ( auto kr : range( kernel_row ) )
                        {
                            std::size_t const offset_kr = offset_c + kr * stride_kr;
                            for ( auto kc : range( kernel_col ) )
                            {
                                std::size_t const offset_kc = offset_kr + kc * stride_kc;
                                for ( auto ch : range( CH ) )
                                {
                                    std::size_t const offset_ch = offset_kc + ch;
                                    *(output.begin()+offset_ch) = pixel_at( input, bs, r, c, ch, padding_row, padding_col );
                                }
                            }
                        }
                    }
                }
            }
        };

        return [row_stride, col_stride, kernel_row, kernel_col, &img2col, &padding]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                []<Tensor Tsor>( Tsor const & tsor, std::vector<Tsor>& context ) noexcept
                {
                    context.resize( 1 );

                    std::size_t padding_row = 0UL;
                    std::size_t padding_col = 0UL;
                    if ( padding == std::string{"same"} )
                    {
                        padding_row = (kernel_row-row_stride+1) >> 1;
                        padding_col = (kernel_col-col_stride+1) >> 1;
                    }

                    img2col( tsor, context[0], padding_row, padding_col );
                    return context[0];
                },
                []<Tensor Tsor>( Tsor const& input, Tsor const& output, Tsor const& grad, std::vector<Tsor>& context ) noexcept
                {

                }
            )( ex );
        };
    }

    auto constexpr conv2d( std::size_t const row_stride, std::size_t const col_stride, std::string const& padding="same" ) noexcept
    {

        // lhs_ex is for one 4D tensor of [BS, R, C, CH]
        // rhs_ex is for NC 4D filter of [1, r, c, CH], thus the shape is [NC, 1, r, c, CH]
        // the output tensor is of shape [BS, .., .., NC]
        //
        // Note: the rhs expression is fixed as a variable, as we need to extract the kernel shape from it
        //
        return [row_stride, col_stride, padding]<Expression Ex, Variable Va>( Expression const& lhs_ex, Va const& rhs_ex ) noexcept
        {
            std::vector<std::size_t> const& shape = rhs_ex.shape();
            better_assert( shape.size() == 4 );
            auto const[kernel_row, kernel_col] = std::make_tuple( shape[1], shape[2] );

            auto lhs_ex_as_col = img2col(row_stride, col_stride, kernel_row, kernel_col, padding)( lhs_ex );
            auto rhs_ex_flatten = flatten( rhs_ex );
            auto flatten_output = rhs_ex_flatten * lhs_ex_as_col;
            auto ans = reshape()( flatten_output );
            return ans;
        };
    }




}//namespace ceras

#endif//IPKVWSJOCMGGVRASCBLPYHFBCHRIVEXYBOMMDAKFAUDFYVYOOOISLRXJNUJKPJEVMLDPRDSNM

