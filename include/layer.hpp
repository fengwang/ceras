#ifndef NLESIGQPSASUTOXPLGXCUHFGGUGYSWLQQFATNISJOSPUFHRORXBNXLSWTYRNSIWJKYFXIQXVN
#define NLESIGQPSASUTOXPLGXCUHFGGUGYSWLQQFATNISJOSPUFHRORXBNXLSWTYRNSIWJKYFXIQXVN

#include "./operation.hpp"

namespace ceras
{
    inline auto Input()
    {
        return place_holder<tensor<float>>{};
    }

    inline auto Conv2D( unsigned long output_channels,std::vector<unsigned long> const& kernel_size, std::vector<unsigned long> const& input_shape, std::string const& padding="valid", std::vector<unsigned long> const& strides={1,1} )
    {
        return [=]<Expression Ex>( Ex const& ex )
        {
            unsigned long const kernel_size_x = kernel_size[0];
            unsigned long const kernel_size_y = kernel_size[1];
            unsigned long const input_channels = input_shape[2];
            unsigned long const input_x = input_shape[0];
            unsigned long const input_y = input_shape[1];
            unsigned long const stride_x = strides[0];
            unsigned long const stride_y = strides[1];
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

}//namespace f

#endif//NLESIGQPSASUTOXPLGXCUHFGGUGYSWLQQFATNISJOSPUFHRORXBNXLSWTYRNSIWJKYFXIQXVN

