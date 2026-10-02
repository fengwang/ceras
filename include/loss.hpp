#ifndef APWVIJWMXHAVXUGYGVNDSEFKTMBKLBMGLSHWUPRPGLFCHUBDRAHGSTDSEDNKOGTIBNQVNLXCD
#define APWVIJWMXHAVXUGYGVNDSEFKTMBKLBMGLSHWUPRPGLFCHUBDRAHGSTDSEDNKOGTIBNQVNLXCD

#include "./operation.hpp"
#include "./tensor.hpp"
#include "./utils/debug.hpp"

namespace ceras
{

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr mean_squared_logarithmic_error( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex )
    {
        return sum_reduce( square( minus(log( value{1.0} + clip(eps)(lhs_ex) ), log( value{1.0} + clip(eps)(rhs_ex) ))) );
    }

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr squared_loss( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex )
    {
        return sum_reduce( square( minus(lhs_ex, rhs_ex)) );
    }

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr mean_squared_error( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex )
    {
        return mean_reduce( square( minus(lhs_ex, rhs_ex)) );
    }

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr mse( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex )
    {
        return mean_squared_error( lhs_ex, rhs_ex );
    }

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr abs_loss( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex )
    {
        return sum_reduce( abs( minus(lhs_ex, rhs_ex)) );
    }

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr mean_absolute_error( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex )
    {
        return mean_reduce( abs( minus(lhs_ex, rhs_ex)) );
    };

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr mae( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex )
    {
        return mean_absolute_error( lhs_ex, rhs_ex );
    };

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr cross_entropy( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex )
    {
        return negative( sum_reduce( hadamard_product( lhs_ex, log(rhs_ex) ) ) );
    }

    namespace
    {
        struct cross_entropy_loss_context
        {
            template<Tensor Tsor> static void validate(Tsor const& target,Tsor const& logits) {
                if(target.shape()!=logits.shape() || target.ndim()!=2 || target.empty() || target.shape()[1]<2)
                    throw std::invalid_argument("cross entropy requires matching nonempty [batch,classes] tensors with at least two classes");
            }
            template<std::floating_point T> auto make_forward(T smoothing) const {
                if(smoothing<0 || smoothing>1) throw std::invalid_argument("label smoothing outside [0,1]");
                return [=]<Tensor Tsor>(Tsor const& target,Tsor const& logits) {
                    validate(target,logits);
                    using V=typename Tsor::value_type;
                    auto batch=target.shape()[0],width=target.shape()[1];V result=0;
                    for(unsigned long row=0;row<batch;++row) {
                        auto first=logits.data()+row*width;
                        V maximum=*std::max_element(first,first+width),total=0;
                        for(unsigned long col=0;col<width;++col)total+=std::exp(first[col]-maximum);
                        V normalizer=std::log(total);
                        for(unsigned long col=0;col<width;++col) {
                            V y=target[row*width+col];V t=(1-smoothing)*y+smoothing*(1-y)/(width-1);
                            result-=t*(first[col]-maximum-normalizer);
                        }
                    }
                    return as_tensor<V,typename Tsor::allocator>(result/batch);
                };
            }
            template<std::floating_point T> auto make_backward(T smoothing) const {
                return [=]<Tensor Tsor>(Tsor const& target,Tsor const& logits,Tsor const&,Tsor const& grad) {
                    validate(target,logits);
                    using V=typename Tsor::value_type;
                    auto batch=target.shape()[0],width=target.shape()[1];V factor=grad[0]/batch;
                    Tsor dy(target.shape()),dx=softmax(logits);
                    for(unsigned long row=0;row<batch;++row) {
                        auto first=logits.data()+row*width;
                        V maximum=*std::max_element(first,first+width),total=0,weight=0;
                        for(unsigned long col=0;col<width;++col) {total+=std::exp(first[col]-maximum);V y=target[row*width+col];weight+=(1-smoothing)*y+smoothing*(1-y)/(width-1);}
                        V normalizer=std::log(total);
                        for(unsigned long col=0;col<width;++col) {
                            auto i=row*width+col;V y=target[i],t=(1-smoothing)*y+smoothing*(1-y)/(width-1);
                            dx[i]=factor*(dx[i]*weight-t);
                            dy[i]=-factor*(1-smoothing-smoothing/(width-1))*(first[col]-maximum-normalizer);
                        }
                    }
                    return std::make_tuple(dy,dx);
                };
            }
        };

    }//anonymous namespace

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr binary_cross_entropy_loss( Lhs_Expression const& ground_truth, Rhs_Expression const& prediction )
    {
        auto ones = ones_like( ground_truth );
        auto error = negative( hadamard_product( ground_truth, log(prediction) ) + hadamard_product( (ones - ground_truth), log(ones - prediction) ) );
        return mean_reduce( error );
    }


    // beware: do not apply softmax activation before this layer, as this loss is softmax+xentropy already
    template < Expression Lhs_Expression, Expression Rhs_Expression, std::floating_point F=float >
    auto constexpr cross_entropy_loss( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex, F label_smoothing_factor=0.0 )
    {
        return make_binary_operator( cross_entropy_loss_context{}.make_forward( label_smoothing_factor ), cross_entropy_loss_context{}.make_backward( label_smoothing_factor ), "CrossEntropyLoss" )( lhs_ex, rhs_ex );
    }

    template < Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr hinge_loss( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex )
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

