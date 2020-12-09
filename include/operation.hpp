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
            output_data_ = forward_action_(  input_data_ );

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
    concept Operation = Operator<T> || Variable<T> || Place_Holder<T>;

    template< Operation Lhs_Operator, Operation Rhs_Operator >
    auto constexpr plus( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
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
                )( lhs_op, rhs_op );
    }

    template< Operation Lhs_Operator, Operation Rhs_Operator >
    auto constexpr operator + ( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return plus( lhs_op, rhs_op );
    }

    template< Operation Lhs_Operator, Operation Rhs_Operator >
    auto constexpr operator * ( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
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
                )( lhs_op, rhs_op );
    }


    template <Operation Op>
    auto constexpr log( Op const& op ) noexcept
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
                )( op );
    };

    template <Operation Op>
    auto constexpr negative( Op const& op ) noexcept
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
                )( op );
    };

    template< Operation Lhs_Operator, Operation Rhs_Operator >
    auto constexpr elementwise_multiply( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
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
                )( lhs_op, rhs_op );
    };

    template< Operation Lhs_Operator, Operation Rhs_Operator >
    auto constexpr hadamard_product( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return elementwise_multiply( lhs_op, rhs_op );
    }

    template <Operation Op>
    auto constexpr sum_reduce( Op const& op ) noexcept
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
                )( op );
    }

    template <Operation Op>
    auto constexpr mean_reduce( Op const& op ) noexcept
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
                )( op );
    }

    template< Operation Lhs_Operator, Operation Rhs_Operator >
    auto constexpr minus( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return plus( lhs_op, negative(rhs_op) );
    }

    template <Operation Op>
    auto constexpr square( Op const& op ) noexcept
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
                )( op );
    }


    template <Operation Op>
    auto constexpr abs( Op const& op ) noexcept
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
                )( op );
    }//;

    template <Operation Op>
    auto constexpr exp( Op const& op ) noexcept
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
                )( op );
    }

    template <typename Float> requires std::floating_point<Float>
    auto constexpr clip( Float lower, Float upper ) noexcept
    {
        return [lower, upper]<Operation Op>( Op const& op ) noexcept
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
                    )( op );
        };
    }

    auto constexpr reshape( std::vector<std::size_t> const& new_shape ) noexcept
    {
        return [&new_shape]<Operation Op>( Op const& op ) noexcept
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
                    )( op );
        };
    }

    template <Operation Op>
    auto constexpr flatten( Op const& op ) noexcept
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
        )( op );
    }

}//namespace ceras

#endif//IPKVWSJOCMGGVRASCBLPYHFBCHRIVEXYBOMMDAKFAUDFYVYOOOISLRXJNUJKPJEVMLDPRDSNM

