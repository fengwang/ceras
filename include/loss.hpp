#ifndef APWVIJWMXHAVXUGYGVNDSEFKTMBKLBMGLSHWUPRPGLFCHUBDRAHGSTDSEDNKOGTIBNQVNLXCD
#define APWVIJWMXHAVXUGYGVNDSEFKTMBKLBMGLSHWUPRPGLFCHUBDRAHGSTDSEDNKOGTIBNQVNLXCD

#include "./operation.hpp"
#include "./tensor.hpp"

namespace ceras
{

#if 0
    template < typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    auto constexpr squared_Loss( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return sum_reduce( square( minus(lhs_op, rhs_op)) );
    }

    template < typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    auto constexpr mean_squared_error( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return mean_reduce( square( minus(lhs_op, rhs_op)) );
    }

    template < typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    auto constexpr mse( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return mean_squared_error( lhs_op, rhs_op );
    }

    template < typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    auto constexpr abs_loss( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return sum_reduce( abs( minus(lhs_op, rhs_op)) );
    }

    template < typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    auto constexpr mean_absolute_error( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return mean_reduce( abs( minus(lhs_op, rhs_op)) );
    };

    template < typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    auto constexpr mae( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return mean_absolute_error( lhs_op, rhs_op );
    };
#endif

    /*
    template < typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    auto constexpr cross_entropy_error( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return negative( sum_reduce( elementwise_multiply( lhs_op, log( clip(1.0e-7, 1.0-1.0e-7)(rhs_op) ) ) ) );
    }
    */

    // softmax + log + nll
    template < typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    auto constexpr cross_entropy_loss( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return make_binary_operator( []<typename T, typename A>( tensor<T,A> const& ground_truth_input, tensor<T,A> const& prediction_input ) noexcept
                                     {
                                        tensor<T,A> sm = softmax( prediction_input );
                                        T ans{0};
                                        for ( auto idx : range( ground_truth_input.size() ) )
                                            ans -= ground_truth_input[idx] * std::log( std::max( static_cast<T>(eps), sm[idx] ) );
                                        return as_tensor<T,A>(ans/(*(ground_truth_input.shape().begin())));
                                     },
                                     // in our implementation, the grad is always 1
                                     []<Tensor Tsor>( Tsor const& ground_truth_input, Tsor const& prediction_input, Tsor const&, Tsor const& ) noexcept
                                     {
                                        Tsor ground_truth_gradient = ground_truth_input;
                                        Tsor sm = softmax( prediction_input ) - ground_truth_input;
                                        return make_tuple( ground_truth_gradient, sm );
                                     }
                )( lhs_op, rhs_op );
    }

    /*
    template < typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    auto constexpr cross_entropy( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return make_binary_operator( []<typename T, typename A>( tensor<T,A> const& ground_truth_input, tensor<T,A> const& prediction_input ) noexcept
                                     {
                                        T ans{0};
                                        for ( auto idx : range( ground_truth_input.size() ) )
                                        {
                                            //ans -= ground_truth_input[idx] * std::log( prediction_input[idx] > 1.0e-10 ? prediction_input[idx] : 1.0e-10 );
                                            ans -= ground_truth_input[idx] * std::log( std::min( 1.0-1.0e-10, std::max( prediction_input[idx], 1.0e-10 ) ) );
                                        }
                                        return as_tensor<T, A>( ans );
                                     },
                                     []<typename T, typename A>( tensor<T,A> const& ground_truth_input, tensor<T,A> const& prediction_input, tensor<T,A> const&, tensor<T,A> const& grad ) noexcept
                                     //[]<typename T, typename A>( tensor<T,A> const& ground_truth_input, tensor<T,A> const& prediction_input, tensor<T,A> const&, tensor<T,A> const grad ) noexcept
                                     {
                                        //for ground_truth branch, just give a random matrix, as no back-propgation will be executed.
                                        tensor<T, A> ground_truth_gradient = ground_truth_input;
                                        //for prediction branch, \partial H / \partial p = gt / p
                                        tensor<T, A> prediction_gradient = deep_copy( prediction_input );
                                        //the back-propagated grad is supposed of shape (1,),
                                        T const g = grad[0];
                                        for_each( prediction_gradient.begin(), prediction_gradient.end(), ground_truth_input.begin(), [g](T& pred, T gt){ pred = - g * gt / ( std::max(1.0e-10, pred) ); } );
                                        return std::make_tuple( ground_truth_gradient, prediction_gradient );
                                     }
                )( lhs_op, rhs_op );
    }
    */

    /*
    template < typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator> && Operation<Rhs_Operator>
    auto constexpr cross_entropy( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
    {
        return make_binary_operator( []<typename T, typename A>( tensor<T,A> const& lhs_tensor, tensor<T,A> const& rhs_tensor ) noexcept
                                     {
                                        better_assert( !has_nan( lhs_tensor ), "cross_entropy_error (0): lhs_tensor has NaN inside!" );

                                        tensor<T,A> rhs = rhs_tensor.deep_copy();
                                        better_assert( is_valid( rhs ), "cross_entropy_error (1): rhs tensor has non-valid inside!" );
                                        rhs -= last_dim_max_reduce( rhs );
                                        better_assert( is_valid( rhs ), "cross_entropy_error (2): rhs tensor has non-valid inside!" );
                                        rhs.map( []( T& x ){ x = std::exp(x); } );
                                        better_assert( is_valid( rhs ), "cross_entropy_error (3): rhs tensor has non-valid inside!" );

                                        rhs /= last_dim_sum_reduce( rhs );
                                        better_assert( is_valid( rhs ), "cross_entropy_error (4): rhs tensor has non-valid inside!" );
                                        clip( rhs, T{1.0e-10}, T{1.0-1.0e-10} );


                                        rhs.map( []( auto& x ){ x = std::log(x); } );
                                        better_assert( is_valid( rhs ), "cross_entropy_error (5): rhs tensor has non-valid inside!" );

                                        auto const& ans = - mean( elementwise_product( lhs_tensor, rhs ) ) * ones<T,A>( {1,} );
                                        better_assert( is_valid( ans ), "cross_entropy_error (6): return tensor has non-valid inside!" );
                                        return ans;
                                     },
                                     []<typename T, typename A>( tensor<T,A> const& lhs_input, tensor<T,A> const& rhs_input, tensor<T,A> const&, tensor<T,A> const grad ) noexcept
                                     {
                                        better_assert( !has_nan( grad ), "backprop: input gradient for cross_entropy_error contains NaN!" );
                                        tensor<T,A> lhs_grad = ones<T>( lhs_input.shape() );

                                        tensor<T,A> rhs = rhs_input.deep_copy();
                                        rhs -= last_dim_max_reduce( rhs );
                                        rhs.map( []( T& x ){ x = std::exp(x); } );

                                        rhs /= last_dim_sum_reduce( rhs );
                                        clip( rhs, T{1.0e-10}, T{1.0-1.0e-10} );

                                        tensor<T,A> rhs_grad = rhs - lhs_input;
                                        better_assert( !has_nan( rhs_grad ), "cross_entropy_error backprop: rhs_grad tensor has NaN inside!" );

                                        return std::make_tuple( lhs_grad, rhs_grad );
                                     }
                )( lhs_op, rhs_op );
    }
    */


}//namespace ceras

#endif//APWVIJWMXHAVXUGYGVNDSEFKTMBKLBMGLSHWUPRPGLFCHUBDRAHGSTDSEDNKOGTIBNQVNLXCD

