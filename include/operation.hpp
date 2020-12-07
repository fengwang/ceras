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
            return [&op]<typename T, typename A>( tensor<T, A> const& grad )
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
            return [op]<typename T, typename A>(tensor<T,A> const& grad)
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


    template< typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    auto constexpr plus( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return make_binary_operator( []<typename T, typename A>( tensor<T,A> const& lhs_tensor, tensor<T,A> const& rhs_tensor ) noexcept
                                     {
                                        better_assert( !has_nan( lhs_tensor ), "forward propagation for operator plus: lhs_tensor contains Nan!" );
                                        better_assert( !has_nan( rhs_tensor ), "forward propagation for operator plus: rhs_tensor contains Nan!" );
                                        return add( lhs_tensor, rhs_tensor );
                                     },
                                     []<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const grad ) noexcept
                                     {
                                        //typedef typename Tsor::value_type value_type;
                                        better_assert( !has_nan( grad ), "backprop: upcoming gradient for operator + contains NaN!" );

                                        auto const& grad_fun = [&grad]( auto const& input )
                                        {
                                            //value_type const batch_size = static_cast<value_type>(grad.size()) / static_cast<value_type>(input.size());
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
                                            //ans /= batch_size;
                                            return ans;
                                        };
                                        return std::make_tuple( grad_fun( lhs_input), grad_fun( rhs_input ) );
                                     }
                )( lhs_op, rhs_op );
    }

    template< typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    auto constexpr operator + ( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return plus( lhs_op, rhs_op );
    }

    template< typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    auto constexpr operator * ( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return make_binary_operator( []<typename T, typename A>( tensor<T,A> const& lhs_tensor, tensor<T,A> const& rhs_tensor ) noexcept
                                     {
                                        //std::cout << "Operator * with lhs:\n" <<  lhs_tensor << std::endl;
                                        //std::cout << "Operator * with rhs:\n" <<  rhs_tensor << std::endl;
                                        better_assert( !has_nan( lhs_tensor ), "forward propagation for operator *: lhs_tensor contains Nan!" );
                                        better_assert( !has_nan( rhs_tensor ), "forward propagation for operator *: rhs_tensor contains Nan!" );
                                        return multiply( lhs_tensor, rhs_tensor );
                                     },
                                     []<typename T, typename A>( tensor<T,A> const& lhs_input, tensor<T,A> const& rhs_input, tensor<T,A> const&, tensor<T,A> const grad ) noexcept
                                     {
                                        better_assert( !has_nan( grad ), "backprop: input gradient for operator * contains NaN!" );
                                        // left branch <-- grad * rhs^T
                                        auto const& g_shape = grad.shape();
                                        auto const[m, n] = std::make_tuple( g_shape[0], g_shape[1] ); // 4, 1
                                        auto const k = *(lhs_input.shape().rbegin()); // 13
                                        tensor<T, A> lhs_grad{ lhs_input.shape() };
                                        gemm( grad.data(), false, rhs_input.data(), true, m, n, k, lhs_grad.data() );

                                        better_assert( !has_nan( lhs_grad ), "backprop: input gradient for operator * -- lhs result contains NaN!" );

                                        // right branch <-- lhs^T * grad
                                        tensor<T,A> rhs_grad{ rhs_input.shape() };
                                        gemm( lhs_input.data(), true, grad.data(), false, k, m, n, rhs_grad.data() );
                                        better_assert( !has_nan( rhs_grad ), "backprop: input gradient for operator * -- rhs result contains NaN!" );


                                        //fix batch_size
                                        //T batch_size = static_cast<T>( *(output.shape().begin()) );
                                        //lhs_grad /= batch_size;
                                        //rhs_grad /= batch_size;

                                        return std::make_tuple( lhs_grad, rhs_grad );
                                        //return std::make_tuple( grad * lhs_input.transpose(), rhs_input.transpose() * grad );
                                     }
                )( lhs_op, rhs_op );
    }


    //static auto constexpr log = []<typename Op>( Op const& op ) noexcept requires Operation<Op>
    template <typename Op> requires Operation<Op>
    auto constexpr log( Op const& op ) noexcept
    {
        return make_unary_operator( []<typename T, typename A>( tensor<T,A> const& input ) noexcept
                                    {
                                        better_assert( !has_nan( input ), "forward propagation for operator log: input contains Nan!" );
                                        auto ans = input.deep_copy();
                                        ans.map( [](T& x){ x = std::log(x); } );
                                        return ans;
                                    },
                                    []<typename T, typename A>( tensor<T,A> const& input, tensor<T,A> const&, tensor<T,A> const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator log contains NaN!" );
                                        auto ans = elementwise_divide(grad, input); // error here
                                        better_assert( !has_nan( ans ), "backprop: result for operator log contains NaN!" );
                                        return ans;
                                    }
                )( op );
    };

    //static auto constexpr negative = []<typename Op>( Op const& op ) noexcept requires Operation<Op>
    template <typename Op> requires Operation<Op>
    auto constexpr negative( Op const& op ) noexcept
    {
        return make_unary_operator( []( auto const& tensor ) noexcept
                                    {
                                        better_assert( !has_nan( tensor ), "forward propagation for operator log: tensor contains Nan!" );
                                        return -tensor;
                                    },
                                    []( auto const&, auto const&, auto const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator negative contains NaN!" );
                                        return -grad;
                                    }
                )( op );
    };

    // elementwise_multiply ==== elementwise_product
    //static auto constexpr elementwise_multiply =[]<typename Lhs_Operator, typename Rhs_Operator>(Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op) noexcept requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    template< typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    auto constexpr elementwise_multiply( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return make_binary_operator( []<typename T, typename A>( tensor<T, A> const& lhs_tensor, tensor<T, A> const& rhs_tensor ) noexcept
                                     {
                                        better_assert( !has_nan( lhs_tensor ), "forward propagation for operator elementwise_multiply: lhs_tensor contains Nan!" );
                                        better_assert( !has_nan( rhs_tensor ), "forward propagation for operator elementwise_multiply: rhs_tensor contains Nan!" );
                                        return elementwise_product( lhs_tensor, rhs_tensor );
                                     },
                                     []<typename T, typename A>( tensor<T,A> const& lhs_input, tensor<T,A> const& rhs_input, tensor<T,A> const&, tensor<T,A> const grad ) noexcept
                                     {
                                        better_assert( !has_nan( grad ), "input gradient for operator elementwise_multiply contains NaN!" );
                                        return std::make_tuple( elementwise_product(grad, rhs_input), elementwise_product(grad, lhs_input) );
                                     }
                )( lhs_op, rhs_op );
    };

    //static auto constexpr sum_reduce = []<typename Op>( Op const& op ) noexcept requires Operation<Op>
    template <typename Op> requires Operation<Op>
    auto constexpr sum_reduce( Op const& op ) noexcept
    {
        return make_unary_operator( []<typename T, typename A>( tensor<T,A> const& tsor ) noexcept
                                    {
                                        better_assert( !has_nan( tsor ), "forward propagation for operator sum_reduce: tensor contains Nan!" );
                                        return reduce_sum( tsor );
                                    },
                                    []<typename T, typename A>( tensor<T,A> const& input, tensor<T,A> const&, tensor<T,A> const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator sum_reduce contains NaN!" );
                                        better_assert( grad.size() == 1, "sum_reduce should only output one value" );
                                        tensor<T,A> ans = ones_like( input );
                                        ans *= grad[0];
                                        return ans;
                                    }
                )( op );
    }

    //static auto constexpr mean_reduce = []<typename Op>( Op const& op ) noexcept requires Operation<Op>
    template <typename Op> requires Operation<Op>
    auto constexpr mean_reduce( Op const& op ) noexcept
    {
        return make_unary_operator( []<typename T, typename A>( tensor<T,A> const& tsor ) noexcept
                                    {
                                        better_assert( !has_nan( tsor ), "forward propagation for operator mean: tensor contains Nan!" );
                                        return reduce_mean( tsor );
                                    },
                                    []<typename T, typename A>( tensor<T,A> const& input, tensor<T,A> const&, tensor<T,A> const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator mean_reduce contains NaN!" );
                                        better_assert( grad.size() == 1, "mean_reduce should only output one value" );
                                        tensor<T,A> ans = ones_like( input );
                                        ans *= grad[0];
                                        std::size_t const batch_size = (input.shape().size() == 1) ? 1 : (*(input.shape().begin()));
                                        ans /= static_cast<T>(batch_size);
                                        return ans;
                                    }
                )( op );
    }

    template< typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    auto constexpr minus( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return plus( lhs_op, negative(rhs_op) );
    }

    /*
    static auto constexpr minus = []<typename Lhs_Operator, typename Rhs_Operator>( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    {
        return plus( lhs_op, negative(rhs_op) );
    };
    */


    //static auto constexpr square = []<typename Op>( Op const& op ) noexcept requires Operation<Op>
    template <typename Op> requires Operation<Op>
    auto constexpr square( Op const& op ) noexcept
    {
        return make_unary_operator( []<typename T, typename A>( tensor<T,A> const& tsor ) noexcept
                                    {
                                        better_assert( !has_nan( tsor ), "forward propagation for operator square: tensor contains Nan!" );
                                        tensor<T,A> ans = tsor.deep_copy();
                                        std::for_each( ans.data(), ans.data() + ans.size(), []( T& v ){ v *= v; } );
                                        return ans;
                                    },
                                    []<typename T, typename A>( tensor<T,A> const& input, tensor<T,A> const&, tensor<T,A> const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator square contains NaN!" );
                                        tensor<T,A> ans = input.deep_copy();
                                        ans *= grad;
                                        ans *= T{2};
                                        return ans;
                                    }
                )( op );
    }


    // use 'ceras::abs' instead of 'abs'
    //static auto constexpr abs = []<typename Op>( Op const& op ) noexcept requires Operation<Op>
    template <typename Op> requires Operation<Op>
    auto constexpr abs( Op const& op ) noexcept
    {
        return make_unary_operator( []<typename T, typename A>( tensor<T,A> const& tsor ) noexcept
                                    {
                                        better_assert( !has_nan( tsor ), "forward propagation for operator abs: tensor contains Nan!" );
                                        tensor<T,A> ans = tsor.deep_copy();
                                        std::for_each( ans.data(), ans.data() + ans.size(), []( T& v ){ v = std::abs(v); } );
                                        return ans;
                                    },
                                    []<typename T, typename A>( tensor<T,A> const& input, tensor<T,A> const&, tensor<T,A> const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator abs contains NaN!" );
                                        tensor<T,A> ans = grad;
                                        for ( auto idx : range( ans.size() ) )
                                            ans[idx] = (input[idx]>T{0}) ? ans[idx] : -ans[idx];
                                        return ans;
                                    }
                )( op );
    }//;

    // use 'ceras::exp' instead of 'exp'
    //static auto constexpr exp = []<typename Op>( Op const& op ) noexcept requires Operation<Op>
    template <typename Op> requires Operation<Op>
    auto constexpr exp( Op const& op ) noexcept
    {
        return make_unary_operator( []<typename T, typename A>( tensor<T,A> const& tsor ) noexcept
                                    {
                                        better_assert( !has_nan( tsor ), "forward propagation for operator exp: tensor contains Nan!" );
                                        tensor<T,A> ans = tsor.deep_copy();
                                        std::for_each( ans.data(), ans.data() + ans.size(), []( T& v ){ v = std::exp(v); } );
                                        return ans;
                                    },
                                    []<typename T, typename A>( tensor<T,A> const&, tensor<T,A> const& output, tensor<T,A> const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator exp contains NaN!" );
                                        tensor<T,A> ans = grad;
                                        grad *= output;
                                        return ans;
                                    }
                )( op );
    }

    template <typename Float> requires std::floating_point<Float>
    auto constexpr clip( Float lower, Float upper ) noexcept
    {
        return [lower, upper]<typename Op>( Op const& op ) noexcept requires Operation<Op>
        {
            return make_unary_operator( [lower, upper]<typename T, typename A>( tensor<T,A> const& tsor ) noexcept
                                        {
                                            better_assert( !has_nan( tsor ), "forward propagation for operator clip: tensor contains Nan!" );
                                            tensor<T,A> ans = tsor.deep_copy();
                                            clip( ans, lower, upper );
                                            return ans;
                                        },
                                        [lower, upper]<typename T, typename A>( tensor<T,A> const& input, tensor<T,A> const&, tensor<T,A> const& grad ) noexcept
                                        {
                                            better_assert( !has_nan( grad ), "input gradient for operator clip contains NaN!" );
                                            tensor<T,A> ans = grad;
                                            for ( auto idx : range( input.size() ) )
                                                ans[idx] = (input[idx] < lower) ? T{0} :
                                                           (input[idx] > upper) ? T{0} :
                                                           ans[idx];
                                            return ans;
                                        }
                    )( op );
        };
    }

}//namespace ceras

#endif//IPKVWSJOCMGGVRASCBLPYHFBCHRIVEXYBOMMDAKFAUDFYVYOOOISLRXJNUJKPJEVMLDPRDSNM
