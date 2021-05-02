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
        return negative( sum_reduce( hadamard_product( lhs_ex, log(rhs_ex) ) ) );
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
                                        auto result = as_tensor<typename Tsor::value_type, typename Tsor::allocator>(ans/(*(ground_truth_input.shape().begin())));
                                        return result;
                                     },
                                     []<Tensor Tsor>( Tsor const& ground_truth_input, Tsor const& prediction_input, [[maybe_unused]]Tsor const& output_data, [[maybe_unused]]Tsor const& grad ) noexcept
                                     {
                                        // in our implementation, the grad is always 1
                                        Tsor ground_truth_gradient = ground_truth_input;
                                        Tsor sm = softmax( prediction_input ) - ground_truth_input;
                                        return std::make_tuple( ground_truth_gradient, sm );
                                     },
                                     "cross_entropy_loss"
                )( lhs_ex, rhs_ex );
    }

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr hinge_loss( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return mean_reduce( maximum( value{0.0f}, value{1.0f} - hadamard_product(lhs_ex, rhs_ex) ) );
    }

    // loss interfaces
    // A loss is an expression. This expression takes two parameters.
    // The first parameter is a place_holder, that will be binded to an tensor.
    // The second parameter is an expression, that will be evaluated to compare with the tensor binded to the first parameter

    inline auto MeanSquaredError = []()
    {
        return []<Expression Ex >( Ex const& output )
        {
            return [=]<Place_Holder Ph>( Ph const& ground_truth )
            {
                return mean_squared_error( ground_truth, output );
            };
        };
    };

    inline auto MeanAbsoluteError = []()
    {
        return []<Expression Ex >( Ex const& output )
        {
            return [=]<Place_Holder Ph>( Ph const& ground_truth )
            {
                return mean_absolute_error( ground_truth, output );
            };
        };
    };

    inline auto Hinge = []()
    {
        return []<Expression Ex >( Ex const& output )
        {
            return [=]<Place_Holder Ph>( Ph const& ground_truth )
            {
                return hinge_loss( ground_truth, output );
            };
        };
    };

    // note: do not apply softmax activation to the last layer of the model, this loss has packaged it
    inline auto CategoricalCrossentropy = []()
    {
        return []<Expression Ex >( Ex const& output )
        {
            return [=]<Place_Holder Ph>( Ph const& ground_truth )
            {
                return cross_entropy_loss( ground_truth, output );
            };
        };
    };



}//namespace ceras

#endif//APWVIJWMXHAVXUGYGVNDSEFKTMBKLBMGLSHWUPRPGLFCHUBDRAHGSTDSEDNKOGTIBNQVNLXCD

