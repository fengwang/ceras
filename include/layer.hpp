#ifndef NLESIGQPSASUTOXPLGXCUHFGGUGYSWLQQFATNISJOSPUFHRORXBNXLSWTYRNSIWJKYFXIQXVN
#define NLESIGQPSASUTOXPLGXCUHFGGUGYSWLQQFATNISJOSPUFHRORXBNXLSWTYRNSIWJKYFXIQXVN

#include "./operation.hpp"
#include "./loss.hpp"
#include "./optimizer.hpp"
#include "./utils/better_assert.hpp"

// try to mimic classes defined in tensorflow.keras

namespace ceras
{

    inline auto Input()
    {
        return place_holder<tensor<float>>{};
    }

    inline auto Conv2D( unsigned long output_channels,std::vector<unsigned long> const& kernel_size, std::vector<unsigned long> const& input_shape, std::string const& padding="valid", std::vector<unsigned long> const& strides={1,1} )
    {
        better_assert( output_channels > 0, "Expecting output_channels larger than 0." );
        better_assert( kernel_size.size() > 0, "Expecting kernel_size at least has 1 elements." );
        better_assert( input_shape.size() ==3, "Expecting input_shape has 3 elements." );
        better_assert( strides.size() > 0, "Expecting strides at least has 1 elements." );
        return [=]<Expression Ex>( Ex const& ex )
        {
            unsigned long const kernel_size_x = kernel_size[0];
            unsigned long const kernel_size_y = kernel_size.size() == 2 ? kernel_size[1] : kernel_size[0];
            //unsigned long const kernel_size_y = kernel_size[1];
            unsigned long const input_channels = input_shape[2];
            unsigned long const input_x = input_shape[0];
            unsigned long const input_y = input_shape[1];
            unsigned long const stride_x = strides[0];
            unsigned long const stride_y = strides.size() == 2 ? strides[1] : strides[0];
            //unsigned long const stride_y = strides[1];
            auto w = variable<tensor<float>>{ glorot_uniform<float>({output_channels, kernel_size_x, kernel_size_y, input_channels}) };
            auto b = variable<tensor<float>>{ zeros<float>({1, 1, output_channels}) };
            return conv2d( input_x, input_y, stride_x, stride_y, 1, 1, padding )( ex, w ) + b;
        };
    }

    inline auto Dense( unsigned long output_size, unsigned long input_size )
    {
        return [=]<Expression Ex>( Ex const& ex )
        {
            auto w = variable<tensor<float>>{ glorot_uniform<float>({input_size, output_size}) };
            auto b = variable<tensor<float>>{ zeros<float>({1, output_size}) };
            return ex * w + b;
        };
    }

    inline auto BatchNormalization( std::vector<unsigned long> const& shape, float threshold = 0.95f )
    {
        return [=]<Expression Ex>( Ex const& ex )
        {
            auto gamma = variable<tensor<float>>{ ones<float>( shape ) };
            auto beta = variable<tensor<float>>{ zeros<float>( shape ) };
            return batch_normalization( threshold )( ex, gamma, beta );
        };
    }


    // losses
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

    //
    // optimizers
    //

    inline auto Adam = []( auto ... args )
    {
        return [=]<Expression Ex>( Ex& loss )
        {
            return adam{loss, args...};
        };
    };

    inline auto SGD = []( auto ... args )
    {
        return [=]<Expression Ex>( Ex& loss )
        {
            return sgd{loss, args...};
        };
    };

    inline auto Adagrad = []( auto ... args )
    {
        return [=]<Expression Ex>( Ex& loss )
        {
            return adagrad{loss, args...};
        };
    };

    inline auto RMSprop = []( auto ... args )
    {
        return [=]<Expression Ex>( Ex& loss )
        {
            return rmsprop{loss, args...};
        };
    };

    inline auto Adadelta = []( auto ... args )
    {
        return [=]<Expression Ex>( Ex& loss )
        {
            return adadelta{loss, args...};
        };
    };

}//namespace f

#endif//NLESIGQPSASUTOXPLGXCUHFGGUGYSWLQQFATNISJOSPUFHRORXBNXLSWTYRNSIWJKYFXIQXVN

