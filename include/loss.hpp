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

    namespace
    {
        struct cross_entropy_loss_context
        {
            template< std::floating_point T >
            auto make_forward( T label_smoothing_factor ) const noexcept
            {
                return [label_smoothing_factor]<Tensor Tsor>( Tsor const& ground_truth_input, Tsor const& prediction_input ) noexcept
                {
                   Tsor sm = softmax( prediction_input );
                   typedef typename Tsor::value_type value_type;
                   typename Tsor::value_type ans{0};
                   unsigned long const n = *(ground_truth_input.shape().rbegin());
                   value_type const _c0 = label_smoothing_factor / (n-1);
                   value_type const _c1 = value_type{1} - label_smoothing_factor;

                   for ( auto idx : range( ground_truth_input.size() ) )
                   {
                       value_type const v = ground_truth_input[idx] > eps ? _c1 : _c0;
                       ans -= v * std::log( std::max( static_cast<typename Tsor::value_type>(eps), sm[idx] ) );
                       //ans -= ground_truth_input[idx] * std::log( std::max( static_cast<typename Tsor::value_type>(eps), sm[idx] ) );
                   }
                   auto result = as_tensor<typename Tsor::value_type, typename Tsor::allocator>(ans/(*(ground_truth_input.shape().begin())));
                   return result;
                };
            }

            template< std::floating_point T >
            auto make_backward( T label_smoothing_factor) const noexcept
            {
                return [=]<Tensor Tsor>( Tsor const& ground_truth_input, Tsor const& prediction_input, [[maybe_unused]]Tsor const& output_data, [[maybe_unused]]Tsor const& grad ) noexcept
                {
                   // In our implementation, the grad is always 1, unless this layer is nested contributing to a combined weighted loss
                   typedef typename Tsor::value_type value_type;
                   value_type const factor = grad[0]; // the shape of grad is {1,}

                   unsigned long const n = *(ground_truth_input.shape().rbegin());
                   value_type const _c0 = label_smoothing_factor / (n-1);
                   value_type const _c1 = value_type{1} - label_smoothing_factor;

                   Tsor ground_truth_gradient = ground_truth_input;

                   //Tsor sm = softmax( prediction_input ) - ground_truth_input;
                   //return std::make_tuple( ground_truth_gradient*factor, sm*factor );

                   Tsor sm = softmax( prediction_input );
                   for ( auto idx : range( ground_truth_input.size() ) )
                   {
                       value_type const v = ground_truth_gradient[idx] > eps ? _c1 : _c0;
                       sm[idx] = factor * (sm[idx] - v );
                   }

                   return std::make_tuple( ground_truth_gradient*factor, sm );
                };
            }

        };//struct cross_entropy_loss_context

    }//anonymous namespace

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr binary_cross_entropy_loss( Lhs_Expression const& ground_truth, Rhs_Expression const& prediction ) noexcept
    {
        auto ones = ones_like( ground_truth );
        auto error = negative( hadamard_product( ground_truth, log(prediction) ) + hadamard_product( (ones - ground_truth), log(ones - prediction) ) );
        return mean_reduce( error );
    }


    // beware: do not apply softmax activation before this layer, as this loss is softmax+xentropy already
    template < Expression Lhs_Expression, Expression Rhs_Expression, std::floating_point F=float >
    auto constexpr cross_entropy_loss( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex, F label_smoothing_factor=0.0 ) noexcept
    {
        return make_binary_operator( cross_entropy_loss_context{}.make_forward( label_smoothing_factor ), cross_entropy_loss_context{}.make_backward( label_smoothing_factor ), "CrossEntropyLoss" )( lhs_ex, rhs_ex );
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

    ///
    /// @brief Computes the mean of squares of errors between labels and predictions.
    ///
    /// \code{.cpp}
    /// auto input = place_holder<tensor<float>>{};
    /// auto v = variable<tensor<float>>{ ones<float>({12, 34}) };
    /// auto output = input * v;
    /// auto m = model{ input, output };
    /// auto cm = m.compile( MeanSquareError(), Adam(128/*batch size*/, 0.01f/*learning rate*/) );
    /// \endcode
    ///
    /// see also #mean_squared_error
    ///
    inline static constexpr auto MeanSquaredError = []()
    {
        return []<Expression Ex >( Ex const& output )
        {
            return [=]<Place_Holder Ph>( Ph const& ground_truth )
            {
                return mean_squared_error( ground_truth, output );
            };
        };
    };

    ///
    /// @brief An alias name of function #MeanSquaredError
    ///
    inline static constexpr auto MSE = []()
    {
        return MeanSquaredError();
    };

    ///
    /// @brief Computes the mean of absolute errors between labels and predictions.
    ///
    /// \code{.cpp}
    /// auto input = place_holder<tensor<float>>{};
    /// auto v = variable<tensor<float>>{ ones<float>({12, 34}) };
    /// auto output = input * v;
    /// auto m = model{ input, output };
    /// auto cm = m.compile( MeanAbsoluteError(), Adam(128/*batch size*/, 0.01f/*learning rate*/) );
    /// \endcode
    ///
    /// see also #mean_absolute_error
    ///
    inline static constexpr auto MeanAbsoluteError = []()
    {
        return []<Expression Ex >( Ex const& output )
        {
            return [=]<Place_Holder Ph>( Ph const& ground_truth )
            {
                return mean_absolute_error( ground_truth, output );
            };
        };
    };


    ///
    /// @brief An alias name of function #MeanAbsoluteError
    ///
    inline static constexpr auto MAE = []()
    {
        return MeanAbsoluteError();
    };



    inline static constexpr auto Hinge = []()
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
    inline static constexpr auto CategoricalCrossentropy = []<std::floating_point F=float>( F label_smoothing_factor = 0.0)
    {
        return [=]<Expression Ex >( Ex const& output )
        {
            return [=]<Place_Holder Ph>( Ph const& ground_truth )
            {
                return cross_entropy_loss( ground_truth, output, label_smoothing_factor );
            };
        };
    };

    inline static constexpr auto CategoricalCrossEntropy = []<std::floating_point F=float>(F label_smoothing_factor = 0.0)
    {
        return CategoricalCrossentropy(label_smoothing_factor);
    };

    inline static constexpr auto BinaryCrossentropy = []()
    {
        return []<Expression Ex >( Ex const& output )
        {
            return [=]<Place_Holder Ph>( Ph const& ground_truth )
            {
                return binary_cross_entropy_loss( ground_truth, output );
            };
        };
    };

    inline static constexpr auto BinaryCrossEntropy = []()
    {
        return BinaryCrossentropy();
    };


}//namespace ceras

#endif//APWVIJWMXHAVXUGYGVNDSEFKTMBKLBMGLSHWUPRPGLFCHUBDRAHGSTDSEDNKOGTIBNQVNLXCD

