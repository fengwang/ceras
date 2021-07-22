#ifndef VRPJCJSTFHSIKPUWMYFOFNFNEVNXXIRHGLOYLMMDVVIPPKMFPBHIEJEOKDTCXFHBKVYUAKJGI
#define VRPJCJSTFHSIKPUWMYFOFNFNEVNXXIRHGLOYLMMDVVIPPKMFPBHIEJEOKDTCXFHBKVYUAKJGI

#include "../includes.hpp"
#include "../tensor.hpp"
#include "./better_assert.hpp"
#include "./range.hpp"

namespace ceras::imageio
{
    //stbir_resize_uint8( input_pixels , in_w , in_h , 0, output_pixels, out_w, out_h, 0, num_channels)
    inline tensor<unsigned char> imresize( tensor<unsigned char> const& input_image, std::vector<unsigned long> const& output_shape )
    {
        std::vector<unsigned long> const& input_shape = input_image.shape();
        tensor<unsigned char> ans{ output_shape };
        if ( input_shape.size() == 3 )
        {
            better_assert( output_shape[2] == input_shape[2], "Expecting same channel for input and output, but the input channel is ", input_shape[2], " and the output channel is ", output_shape[2] );
            stbir_resize_uint8( input_image.data(), input_shape[0], input_shape[1], 0, ans.data(), output_shape[0], output_shape[1], 0, input_shape[2] );
        }
        else
        {
            stbir_resize_uint8( input_image.data(), input_shape[0], input_shape[1], 0, ans.data(), output_shape[0], output_shape[1], 0, 2 ); // gray
        }

        return ans;
    }

    inline tensor<unsigned char> imread( std::string const& path )
    {
        int width, height, channels;
        unsigned char *img = stbi_load( path.c_str(), &width, &height, &channels, 0);
        if ( img == nullptr ) return {};

        tensor<unsigned char> ans{ {static_cast<unsigned long>(width), static_cast<unsigned long>(height), static_cast<unsigned long>(channels)} };
        std::copy_n( img, width*height*channels, ans.begin() );
        stbi_image_free(img);
        return ans;
    }

    template< Tensor Tsor >
    inline bool direct_imwrite( std::string const& path, Tsor const& image )
    {
        better_assert( image.ndim() == 2 || image.ndim() == 3, "Expecting an input tensor of 2D or 3D, but got ", image.ndim() );

        // extract w, h, comp
        auto const& shape = image.shape();
        int const w = shape[0];
        int const h = shape[1];
        int const ch = shape.size() == 2 ? 1 : shape[2];
        better_assert( (ch == 1) || (ch == 3) || (ch == 4), " got a wrong number of channels: ", ch );

        // setting up extension
        std::string full_path = path;
        std::string file_extension=full_path.substr(full_path.find_last_of('.'));
        if ( file_extension.empty() )
        {
            full_path += std::string{".png"};
            file_extension = std::string{".png"};
        }

        if ( file_extension == std::string{".png"} )
            return 0 != stbi_write_png( full_path.c_str(), w, h, ch, reinterpret_cast<const void*>( image.data() ), 0 );

        if ( file_extension == std::string{".bmp"} )
            return 0 != stbi_write_bmp( full_path.c_str(), w, h, ch, reinterpret_cast<const void*>( image.data() ) );

        if ( file_extension == std::string{".tga"} )
            return 0 != stbi_write_tga( full_path.c_str(), w, h, ch, reinterpret_cast<const void*>( image.data() ) );

        if ( file_extension == std::string{".jpg"} || file_extension == std::string{".jpeg"} )
            return 0 != stbi_write_jpg( full_path.c_str(), w, h, ch, reinterpret_cast<const void*>( image.data() ), 100 );

        if ( file_extension == std::string{".hdr"} )
            return 0 != stbi_write_hdr( full_path.c_str(), w, h, ch, reinterpret_cast<const float*>( image.data() ) );

        // reaching here, the extension is unknown, png is the last chance
        full_path += std::string{".png"};
        return 0 != stbi_write_png( full_path.c_str(), w, h, ch, reinterpret_cast<const void*>( image.data() ), 0 );
    }

    template< Tensor Tsor >
    inline bool imwrite( std::string const& path, Tsor const& image )
    {
        typedef typename Tsor::value_type value_type;
        if (sizeof(value_type) == 1)
        {
            return direct_imwrite( path, image );
        }

        value_type const mx = amax( image );
        value_type const mn = amin( image );

        tensor<std::uint8_t> image_{ image.shape() };
        for_each( image.begin(), image.end(), image_.begin(), [mx, mn]( value_type x, std::uint8_t& v ) { v = static_cast<std::uint8_t>( 255.0 * (x - mn) / (mx-mn+1.0e-10) ); } );
        return direct_imwrite( path, image_ );
    }

}//namespace ceras::imageio

#endif//VRPJCJSTFHSIKPUWMYFOFNFNEVNXXIRHGLOYLMMDVVIPPKMFPBHIEJEOKDTCXFHBKVYUAKJGI

