#ifndef APWVIJWMXHAVXUGYGVNDSEFKTMBKLBMGLSHWUPRPGLFCHUBDRAHGSTDSEDNKOGTIBNQVNLXCD
#define APWVIJWMXHAVXUGYGVNDSEFKTMBKLBMGLSHWUPRPGLFCHUBDRAHGSTDSEDNKOGTIBNQVNLXCD

#include "./operation.hpp"
#include "./tensor.hpp"
#include "./utils/debug.hpp"

namespace ceras
{

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr mean_squared_logarithmic_error( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return sum_reduce( square( minus(lhs_ex, rhs_ex)) );
    }

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
        //return negative( sum_reduce( elementwise_multiply( lhs_ex, log(rhs_ex) ) ) );
        return negative( sum_reduce( hadamard_product( lhs_ex, log(rhs_ex) ) ) );
    }

    // beware: do not apply softmax activation before this layer, as this loss is softmax+xentropy already
    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr cross_entropy_loss( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        if constexpr ( std::is_same_v<Lhs_Expression, place_holder<tensor<float>>> )
        {
            //debug_print( "corss enxtropy loss created with lhs being place holder: ", lhs_ex.id_ );
        }
        if constexpr ( std::is_same_v<Rhs_Expression, place_holder<tensor<float>>> )
        {
            //debug_print( "corss enxtropy loss created with rhs being place holder: ", rhs_ex.id_ );
        }
        return make_binary_operator( []<Tensor Tsor>( Tsor const& ground_truth_input, Tsor const& prediction_input ) noexcept
                                     {
                                        debug_print( "Cross entropy loss forwarded with lhs tensor ", ground_truth_input.id_, " and rhs tensor id ", prediction_input.id_ );
                                        Tsor sm = softmax( prediction_input );
                                        typename Tsor::value_type ans{0};
                                        for ( auto idx : range( ground_truth_input.size() ) )
                                            ans -= ground_truth_input[idx] * std::log( std::max( static_cast<typename Tsor::value_type>(eps), sm[idx] ) );
                                        auto result = as_tensor<typename Tsor::value_type, typename Tsor::allocator>(ans/(*(ground_truth_input.shape().begin())));
                                        debug_print( "Cross entropy loss returns a result tensor with id ", result.id_, ", and the actual ans is ", result[0] );
                                        return result;
                                     },
                                     // in our implementation, the grad is always 1
                                     // lhs_input_data_, rhs_input_data_, output_data_, grad
                                     []<Tensor Tsor>( Tsor const& ground_truth_input, Tsor const& prediction_input, [[maybe_unused]]Tsor const& output_data, [[maybe_unused]]Tsor const& grad ) noexcept
                                     {
                                        debug_print( "Cross entropy loss backwarede, with input tensor ", ground_truth_input.id_, ", prediction_input ", prediction_input.id_, ", output_data ", output_data.id_, " and grad ", grad.id_ );
                                        debug_print( "the size for ground_truth_input is ", ground_truth_input.size() );
                                        debug_print( "the size for prediction_input is ", prediction_input.size() );
                                        debug_print( "the size for output_data is ", output_data.size() );
                                        debug_print( "the size for grad is ", grad.size() );
                                        Tsor ground_truth_gradient = ground_truth_input;
                                        Tsor sm = softmax( prediction_input ) - ground_truth_input;
                                        return std::make_tuple( ground_truth_gradient, sm );
                                     },
                                     "cross_entropy_loss"
                )( lhs_ex, rhs_ex );
    }

}//namespace ceras

#endif//APWVIJWMXHAVXUGYGVNDSEFKTMBKLBMGLSHWUPRPGLFCHUBDRAHGSTDSEDNKOGTIBNQVNLXCD

