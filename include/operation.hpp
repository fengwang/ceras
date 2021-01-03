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

        unary_operator( Operator const& op, Forward_Action const& forward_action, Backward_Action const& backward_action ) noexcept :
            op_{operator_type_wrapper{}(op)}, forward_action_{ forward_action }, backward_action_{ backward_action } {}

        // update output_data, reset gradient
        auto forward()// const
        {
            input_data_ = forward_wrapper{}( op_ );
            output_data_ = forward_action_( input_data_ );

            return output_data_;
        }

        // update gradient
        template< typename T, typename A >
        void backward( tensor<T,A> const& grad )
        {
            auto const& current_gradient = backward_action_( input_data_, output_data_, grad );
            backward_wrapper{}( op_ )( current_gradient );
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

        binary_operator( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op, Forward_Action const& forward_action, Backward_Action const& backward_action ) noexcept :
            lhs_op_{operator_type_wrapper{}(lhs_op)}, rhs_op_{operator_type_wrapper{}(rhs_op)}, forward_action_{ forward_action }, backward_action_{ backward_action } {}

        auto forward()// const
        {
            lhs_input_data_ = forward_wrapper{}( lhs_op_ );
            rhs_input_data_ = forward_wrapper{}( rhs_op_ );

            output_data_ = forward_action_( lhs_input_data_, rhs_input_data_ );
            return output_data_;
        }

        template< typename T, typename A >
        void backward( tensor<T,A> const& grad )
        {
            auto const& [current_gradient_lhs, current_gradient_rhs] = backward_action_( lhs_input_data_, rhs_input_data_, output_data_, grad );
            backward_wrapper{}( lhs_op_ )( current_gradient_lhs );
            backward_wrapper{}( rhs_op_ )( current_gradient_rhs );

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

    auto constexpr swap_axes( std::size_t axis_1, std::size_t axis_2 ) noexcept
    {
        return [=]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [=]<Tensor Tsor>( Tsor const& tsor ) noexcept
                {
                    return tsor;//TODO: fix
                },
                [=]<Tensor Tsor>( Tsor const& input, Tsor const& output, Tsor const& grad ) noexcept
                {
                    return grad;//TODO: fix
                }
            )( ex );
        };
    }

    template< Expression Ex >
    auto constexpr transpose( Ex const& ex ) noexcept
    {
        return make_unary_operator
        (
            []<Tensor Tsor>( Tsor const& tsor ) noexcept
            {
                better_assert( tsor.ndim() == 2, "Expecting 2D tensor, but got dimensions ", tsor.ndim() );

                typedef typename Tsor::value_type value_type;

                std::vector<std::size_t> const shape = tsor.shape();
                auto const[row, col] = std::make_tuple( shape[0], shape[1] );
                view_2d<value_type> v_in{ tsor.data(), row, col };

                Tsor ans{ {col, row} }; // TODO: optimize it out with shared_ptr
                view_2d<value_type> v_out{ ans.data(), col, row };

                for ( auto r : range( row ) )
                    for ( auto c : range( col ) )
                        v_out[c][r] = v_in[r][c];

                return ans;
            },
            []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
            {
                typedef typename Tsor::value_type value_type;

                std::vector<std::size_t> const shape = grad.shape();
                auto const[row, col] = std::make_tuple( shape[0], shape[1] );
                view_2d<value_type> v_in{ grad.data(), row, col };

                Tsor back_ans{ {col, row} }; // TODO: optimize it out with shared_ptr
                view_2d<value_type> v_out{ back_ans.data(), col, row };

                for ( auto r : range( row ) )
                    for ( auto c : range( col ) )
                        v_out[c][r] = v_in[r][c];

                return back_ans;
            }
        )( ex );
    }

    auto inline img2col( std::size_t const row_kernel, std::size_t col_kernel=-1,
                         std::size_t const row_padding=0, std::size_t col_padding=0,
                         std::size_t const row_stride=1, std::size_t const col_stride=1,
                         std::size_t const row_dilation=1, std::size_t const col_dilation=1 ) noexcept
    {
        if ( col_kernel == (std::size_t)-1 ) col_kernel = row_kernel;

        std::shared_ptr<std::vector<std::uint32_t>> s_index_record = std::make_shared<std::vector<std::uint32_t>>(); // col_img[idx] = img[index_record[idx]]  -- (-1) for zero padding

        auto img2col_forward = [s_index_record]<Tensor Tsor>
        (
            Tsor const& input_img, Tsor& output_col_mat,
            std::size_t kernel_row, std::size_t kernel_col,
            std::size_t padding_row, std::size_t padding_col,
            std::size_t stride_row, std::size_t stride_col,
            std::size_t dilation_row, std::size_t dilation_col
        ) noexcept
        {
            debug_print( "img2col_forward with kernel_row=", kernel_row, ", kernel_col=", kernel_col, ", stride_row=", stride_row, ", stride_col=", stride_col, ", dilation_row=", dilation_row, ", dilation_col=", dilation_col );

            typedef typename Tsor::value_type value_type;
            std::vector<std::uint32_t>& index_record = *s_index_record; //32 bit should be enough for memory address offeset

            std::vector<std::size_t> input_shape = input_img.shape();
            better_assert( input_shape.size() == 4, "Expecting a 4D tensor." );
            auto const [BS, R, C, CH] = std::make_tuple( input_shape[0], input_shape[1], input_shape[2], input_shape[3] );

            debug_print( "img2col_forward: Bs=", BS, " R=", R, " C=", C, " CH=", CH );

            std::size_t const output_row = ( R + 2 * padding_row - ( dilation_row * (kernel_row - 1) + 1 ) ) / stride_row + 1;
            std::size_t const output_col = ( C + 2 * padding_col - ( dilation_col * (kernel_col - 1) + 1 ) ) / stride_col + 1;
            std::size_t const output_column_matrix_row = kernel_row * kernel_col * CH;
            std::size_t const output_column_matrix_col = BS * output_row * output_col;

            debug_print( "img2col_forward: output_row=", output_row, ", output_col=", output_col, ", output_column_matrix_row=", output_column_matrix_row, ", output_column_matrix_col=", output_column_matrix_col );

            output_col_mat.resize( {output_column_matrix_row, output_column_matrix_col} );

            debug_print( "img2col_forward: outptu_col_mat resize with output_column_matrix_row=", output_column_matrix_row, ", output_column_matrix_col=", output_column_matrix_col );

            if ( index_record.size() != output_column_matrix_row * output_column_matrix_col ) // first-run?
            {
                debug_print( "index_record has not yet been filled, trying to fill it!" );
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

        auto img2col_backward = [s_index_record]<Tensor Tsor>( Tsor const& input, Tsor const& output, Tsor const& grad, Tsor& ans ) noexcept
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
                    std::any& output_cache_tsor = *output_cache;
                    if ( !output_cache_tsor.has_value() )
                        output_cache_tsor = Tsor{};
                    Tsor& output = std::any_cast<Tsor&>(output_cache_tsor);
                    //Tsor output;
                    img2col_forward( tsor, output, row_kernel, col_kernel, row_padding, col_padding, row_stride, col_stride, row_dilation, col_dilation );
                    return Tsor{output};
                },
                [=]<Tensor Tsor>( Tsor const& input, Tsor const& output, Tsor const& grad ) noexcept
                {
                    std::any& back_grad_cache_tsor = *back_grad_cache;
                    if ( !back_grad_cache_tsor.has_value() )
                        back_grad_cache_tsor = Tsor{};
                    Tsor& back_grad = std::any_cast<Tsor&>( back_grad_cache_tsor );
                    //Tsor back_grad;
                    img2col_backward( input, output, grad, back_grad );
                    return Tsor{back_grad};
                }
            )( ex );
        };
    }

    auto inline conv2d( std::size_t const row_stride, std::size_t const col_stride, std::size_t const row_dilation, std::size_t const col_dilation, std::string const& padding="same" ) noexcept
    {

        // lhs_ex is for one 4D tensor of [BS, R, C, CH]
        // rhs_ex is for NC 4D filter of [1, r, c, CH], thus the shape is [NC, r, c, CH]
        // the output tensor is of shape [BS, .., .., NC]
        //
        // Note: the rhs expression is fixed as a variable, as we need to extract the kernel shape from it
        //
        return [row_stride, col_stride, row_dilation, col_dilation, padding]<Expression Ex, Variable Va>( Ex const& lhs_ex, Va const& rhs_ex ) noexcept
        {
            std::vector<std::size_t> const& shape = rhs_ex.shape();
            better_assert( shape.size() == 4 );
            auto const[new_channel, row_kernel, col_kernel, channel] = std::make_tuple( shape[0], shape[1], shape[2], shape[3] );

            std::size_t row_padding = 0;
            std::size_t col_padding = 0;
            if ( padding == "same" )
            {
                row_padding = (row_kernel + (row_kernel - 1) * (row_dilation - 1) - row_stride) >> 1;
                col_padding = (col_kernel + (col_kernel - 1) * (col_dilation - 1) - col_stride) >> 1;
            }
            if ( padding == "transposed" )
            {
                row_padding = row_kernel - 1;
                col_padding = col_kernel - 1;
            }

            auto lhs_ex_as_col = img2col(row_kernel, col_kernel, row_padding, col_padding, row_stride, col_stride, row_dilation, col_dilation)( lhs_ex ); // [BS, R, C, CH] ==> [r*c*CH, BS*new_row*new_col]
            auto rhs_ex_flatten = reshape({row_kernel*col_kernel*channel})( rhs_ex ); // [NC, r, c, CH] ==> [NC, r*c*CH]
            auto flatten_output = rhs_ex_flatten * lhs_ex_as_col; // [NC, BS * new_row * new_col]
            auto tr_output = transpose( flatten_output ); // [BS*new_row*new_col, NC]
            std::size_t const new_row = 1;//TODO: fixme
            std::size_t const new_col = 1;//TODO: fixme
            auto ans = reshape({new_row, new_col, new_channel})( tr_output );
            return ans;
        };
    }



}//namespace ceras

#endif//IPKVWSJOCMGGVRASCBLPYHFBCHRIVEXYBOMMDAKFAUDFYVYOOOISLRXJNUJKPJEVMLDPRDSNM

