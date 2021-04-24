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

    //
    // TODO: PReLU
    // TODO: AveragePooling2D
    // TODO: Dropout
    // TODO: Reshape
    // TODO: Flatten
    // TODO: UpSampling2D
    //








}//namespace f

#endif//NLESIGQPSASUTOXPLGXCUHFGGUGYSWLQQFATNISJOSPUFHRORXBNXLSWTYRNSIWJKYFXIQXVN

