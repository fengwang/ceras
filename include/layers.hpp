#ifndef NLESIGQPSASUTOXPLGXCUHFGGUGYSWLQQFATNISJOSPUFHRORXBNXLSWTYRNSIWJKYFXIQXVN
#define NLESIGQPSASUTOXPLGXCUHFGGUGYSWLQQFATNISJOSPUFHRORXBNXLSWTYRNSIWJKYFXIQXVN

#include "./operation.hpp"

namespace ceras
{

    constexpr inline auto Conv2D( unsigned long output_channels,std::vector<unsigned long> const& kernel_size, std::vector<unsigned long> const& input_shape )
    {
        return [=]<Expression Ex>( Ex const& ex )
        {
            unsigned long const kernel_size_x = kernel_size[0];
            unsigned long const kernel_size_y = kernel_size[1];
            unsigned long const input_channels = input_shape[2];
            unsigned long const input_x = input_shape[0];
            unsigned long const input_y = input_shape[1];
            auto w = variable<tensor<float>>{ glorot_uniform<float>({output_channels, kernel_size_x, kernel_size_y, input_channels}) };
            auto b = variable<tensor<float>>{ zeros<float>({1, 1, output_channels}) };
            return conv2d( input_x, input_y, 1, 1, 1, 1, "same" )( ex, w ) + b;
        };
    }

    constexpr inline auto Dense( unsigned long output_size, unsigned long input_size )
    {
        return [=]<Expression Ex>( Ex const& ex )
        {
            auto w = variable<tensor_type>{ glorot_uniform<float>({input_size, output_size}) };
            auto b = variable<tensor_type>{ zeros<float>({1, output_size}) };
            return ex * w + b;
        };
    }


}//namespace f

#endif//NLESIGQPSASUTOXPLGXCUHFGGUGYSWLQQFATNISJOSPUFHRORXBNXLSWTYRNSIWJKYFXIQXVN

