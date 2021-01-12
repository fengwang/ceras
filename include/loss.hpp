#ifndef APWVIJWMXHAVXUGYGVNDSEFKTMBKLBMGLSHWUPRPGLFCHUBDRAHGSTDSEDNKOGTIBNQVNLXCD
#define APWVIJWMXHAVXUGYGVNDSEFKTMBKLBMGLSHWUPRPGLFCHUBDRAHGSTDSEDNKOGTIBNQVNLXCD

#include "./operation.hpp"
#include "./tensor.hpp"

namespace ceras
{

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr squared_loss( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return sum_reduce( square( minus(lhs_ex, rhs_ex)) );
    }

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr mean_squared_error( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return mean_reduce( square( minus(lhs_ex, rhs_ex)) );
    }

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr mse( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return mean_squared_error( lhs_ex, rhs_ex );
    }

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr abs_loss( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return sum_reduce( abs( minus(lhs_ex, rhs_ex)) );
    }

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr mean_absolute_error( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return mean_reduce( abs( minus(lhs_ex, rhs_ex)) );
    };

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr mae( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return mean_absolute_error( lhs_ex, rhs_ex );
    };

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr cross_entropy( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return negative( sum_reduce( elementwise_multiply( lhs_ex, log(rhs_ex) ) ) );
    }

    // beware: do not apply softmax activation before this layer, as this loss is softmax+xentropy already
    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr cross_entropy_loss( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return make_binary_operator( []<Tensor Tsor>( Tsor const& ground_truth_input, Tsor const& prediction_input ) noexcept
                                     {
                                        Tsor sm = softmax( prediction_input );
                                        typename Tsor::value_type ans{0};
                                        for ( auto idx : range( ground_truth_input.size() ) )
                                            ans -= ground_truth_input[idx] * std::log( std::max( static_cast<typename Tsor::value_type>(eps), sm[idx] ) );
                                        return as_tensor<typename Tsor::value_type, typename Tsor::allocator>(ans/(*(ground_truth_input.shape().begin())));
                                     },
                                     // in our implementation, the grad is always 1
                                     []<Tensor Tsor>( Tsor const& ground_truth_input, Tsor const& prediction_input, Tsor const&, Tsor const& ) noexcept
                                     {
                                        Tsor ground_truth_gradient = ground_truth_input;
                                        Tsor sm = softmax( prediction_input ) - ground_truth_input;
                                        return std::make_tuple( ground_truth_gradient, sm );
                                     }
                )( lhs_ex, rhs_ex );
    }

}//namespace ceras

#endif//APWVIJWMXHAVXUGYGVNDSEFKTMBKLBMGLSHWUPRPGLFCHUBDRAHGSTDSEDNKOGTIBNQVNLXCD

