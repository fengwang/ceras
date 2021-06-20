#ifndef NLESIGQPSASUTOXPLGXCUHFGGUGYSWLQQFATNISJOSPUFHRORXBNXLSWTYRNSIWJKYFXIQXVN
#define NLESIGQPSASUTOXPLGXCUHFGGUGYSWLQQFATNISJOSPUFHRORXBNXLSWTYRNSIWJKYFXIQXVN

#include "./operation.hpp"
#include "./activation.hpp"
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

    ///
    /// @brief 2D convolution layer.
    /// @param output_channels Dimensionality of the output space.
    /// @param kernel_size The height and width of the convolutional window.
    /// @param input_shape Dimensionality of the input shape.
    /// @param padding `valid` or `same`. `valid` suggests no padding. `same` suggests zero padding. Defaults to `valid`.
    /// @param strides The strides along the height and width direction. Defaults to `(1, 1)`.
    /// @param dilations The dialation along the height and width direction. Defaults to `(1, 1)`.
    /// @param use_bias Wether or not use a bias vector. Defaults to `true`.
    ///
    /// Example code:
    ///
    /// \code{.cpp}
    /// auto x = Input{};
    /// auto y = Conv2D( 32, {3, 3}, {28, 28, 1}, "same" )( x );
    /// auto z = Flatten()( y );
    /// auto u = Dense( 10, 28*28*32 )( z );
    /// auto m = model{ x, u };
    /// \endcode
    ///
    inline auto Conv2D( unsigned long output_channels, std::vector<unsigned long> const& kernel_size,
                        std::vector<unsigned long> const& input_shape, std::string const& padding="valid",
                        std::vector<unsigned long> const& strides={1,1}, std::vector<unsigned long> const& dilations={1, 1}, bool use_bias=true )
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
            unsigned long const dilation_row = dilations[0];
            unsigned long const dilation_col = dilations.size() == 2 ? dilations[1] : dilations[0];
            //unsigned long const stride_y = strides[1];
            auto w = variable<tensor<float>>{ glorot_uniform<float>({output_channels, kernel_size_x, kernel_size_y, input_channels}) };
            auto b = variable<tensor<float>>{ zeros<float>({1, 1, output_channels}), use_bias };
            return conv2d( input_x, input_y, stride_x, stride_y, dilation_row, dilation_col, padding )( ex, w ) + b;
        };
    }

    ///
    /// @brief Densly-connected layer.
    ///
    /// @param output_size Dimensionality of output shape. The output shape is `(batch_size, output_size)`.
    /// @param input_size Dimensionality of input shape. The input shape is `(batch_size, input_size)`.
    /// @param use_bias Using a bias vector or not. Defaults to `true`.
    ///
    /// Example code:
    ///
    /// \code{.cpp}
    /// auto x = Input{};
    /// auto y = Dense( 10, 28*28 )( x );
    /// auto m = model{ x, y };
    /// \endcode
    ///
    inline auto Dense( unsigned long output_size, unsigned long input_size, bool use_bias=true )
    {
        return [=]<Expression Ex>( Ex const& ex )
        {
            auto w = variable<tensor<float>>{ glorot_uniform<float>({input_size, output_size}) };
            auto b = variable<tensor<float>>{ zeros<float>({1, output_size}), use_bias }; // if use_baias, then b is trainable; otherwise, non-trainable.
            return ex * w + b;
        };
    }

    inline auto BatchNormalization( std::vector<unsigned long> const& shape, float threshold = 0.95f )
    {
        return [=]<Expression Ex>( Ex const& ex )
        {
            unsigned long const last_dim = *(shape.rbegin());
            auto gamma = variable<tensor<float>>{ ones<float>( {last_dim, }  ) };
            auto beta = variable<tensor<float>>{ zeros<float>( {last_dim, } ) };
            return batch_normalization( threshold )( ex, gamma, beta );
        };
    }

    inline auto BatchNormalization( float threshold, std::vector<unsigned long> const& shape )
    {
        return BatchNormalization( shape, threshold );
    }

#if 0
    // TODO: fix this layer
    inline auto LayerNormalization( std::vector<unsigned long> const& shape )
    {
        return [=]<Expression Ex>( Ex const& ex )
        {
            unsigned long const last_dim = *(shape.rbegin());
            auto gamma = variable<tensor<float>>{ ones<float>( {last_dim, }  ) };
            auto beta = variable<tensor<float>>{ zeros<float>( {last_dim, } ) };
            return layer_normalization()( ex, gamma, beta );
        };
    }
#endif

    ///
    /// Layer that concatenates two layers.
    /// @param axis The concatenation axis. Default to the last channel.
    ///
    /// Example usage:
    /// @code
    /// auto l1 = variable{ tensor<float>{ {12, 11, 3} } };
    /// auto l2 = variable{ tensor<float>{ {12, 11, 4} } };
    /// auto l12 = Concatenate()( l1, l2 ); // should be of shape (12, 11, 7)
    /// @endcode
    ///
    inline auto Concatenate(unsigned long axis = -1) noexcept
    {
        return [=]<Expression Lhs_Expression, Expression Rhs_Expression>( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
        {
            return concatenate( axis )( lhs_ex, rhs_ex );
        };
    }

    ///
    /// Layer that adds two layers
    ///
    /// Example usage:
    /// @code
    /// auto input = Input(); // (16, )
    /// auto x1 = Dense( 8, 16 )( input );
    /// auto x2 = Dense( 8, 16 )( input );
    /// auto x3 = Add()( x1, x2 ); // equivalent to `x1 + x2`
    /// auto m = model{ input, x3 };
    /// @endcode
    ///
    inline auto Add() noexcept
    {
        return []<Expression Lhs_Expression, Expression Rhs_Expression>( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
        {
            return lhs_ex + rhs_ex;
        };
    }


    ///
    /// Layer that subtracts two layers
    ///
    /// Example usage:
    /// @code
    /// auto input = Input(); // (16, )
    /// auto x1 = Dense( 8, 16 )( input );
    /// auto x2 = Dense( 8, 16 )( input );
    /// auto x3 = Subtract()( x1, x2 ); // equivalent to `x1 - x2`
    /// auto m = model{ input, x3 };
    /// @endcode
    ///
    inline auto Subtract() noexcept
    {
        return []<Expression Lhs_Expression, Expression Rhs_Expression>( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
        {
            return lhs_ex - rhs_ex;
        };
    }

    ///
    /// Layer that elementwise multiplies two layers
    ///
    /// Example usage:
    /// @code
    /// auto input = Input(); // (16, )
    /// auto x1 = Dense( 8, 16 )( input );
    /// auto x2 = Dense( 8, 16 )( input );
    /// auto x3 = Multiply()( x1, x2 ); // equivalent to `elementwise_multiply(x1, x2)`
    /// auto m = model{ input, x3 };
    /// @endcode
    ///
    inline auto Multiply() noexcept
    {
        return []<Expression Lhs_Expression, Expression Rhs_Expression>( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
        {
            return hadamard_product( lhs_ex, rhs_ex );
        };
    }

    ///
    /// Rectified Linear Unit activation function.
    ///
    template< Expression Ex >
    inline auto ReLU( Ex const& ex ) noexcept
    {
        return relu( ex );
    }

    ///
    /// Softmax activation function.
    ///
    inline auto Softmax() noexcept
    {
        return []< Expression Ex >( Ex const& ex ) noexcept
        {
            return softmax( ex );
        };
    }


    ///
    /// leaky relu activation function.
    ///
    template< typename T = float >
    inline auto LeakyReLU( T const factor=0.2 ) noexcept
    {
        return leaky_relu( factor );
    }

    ///
    /// Exponential Linear Unit.
    ///
    template< typename T = float >
    inline auto ELU( T const factor=0.2 ) noexcept
    {
        return elu( factor );
    }


    ///
    /// Reshapes inputs into the given shape.
    ///
    inline auto Reshape( std::vector<unsigned long> const& new_shape, bool include_batch_flag=true ) noexcept
    {
        return reshape( new_shape, include_batch_flag );
    }

    ///
    /// Flattens the input. Does not affect the batch size.
    ///
    inline auto Flatten() noexcept
    {
        return []<Expression Ex>( Ex const& ex ) noexcept
        {
            return flatten( ex );
        };
    }

    ///
    /// Max pooling operation for 2D spatial data.
    ///
    inline auto MaxPooling2D( unsigned long stride ) noexcept
    {
        return max_pooling_2d( stride );
    }

    ///
    /// Upsampling layer for 2D inputs.
    ///
    inline auto UpSampling2D( unsigned long stride ) noexcept
    {
        return up_sampling_2d( stride );
    }

    ///
    /// Applies Dropout to the input.
    ///
    template< typename T >
    inline auto Dropout( T factor ) noexcept
    {
        return drop_out( factor );
    }

    ///
    /// Average pooling operation for spatial data.
    ///
    inline auto AveragePooling2D( unsigned long stride ) noexcept
    {
        return average_pooling_2d( stride );
    }

    //
    // TODO: PReLU
    //








}//namespace f

#endif//NLESIGQPSASUTOXPLGXCUHFGGUGYSWLQQFATNISJOSPUFHRORXBNXLSWTYRNSIWJKYFXIQXVN

