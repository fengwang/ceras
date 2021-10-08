#include "../include/tensor.hpp"
#include "../include/utils/imageio.hpp"
#include <iostream>

int main()
{
    {
        unsigned long n = 1024;
        using namespace ceras;
        auto const& a = ones<double>( {n, n} ) * 100.0;
        auto const& b = poisson( a );
        auto const& c = b.as_type<std::uint8_t>();
        imageio::imwrite( "./test/poisson.png", c );
    }
    {
        using namespace ceras;
        auto const& lena = imageio::imread( "./dataset/lena.png" );
        auto n_lena = poisson( lena.as_type<long>() );
        clip( n_lena, 0, 255 );
        imageio::imwrite( "./test/lena_poisson.png", n_lena.as_type<std::uint8_t>() );
    }

    return 0;
}

