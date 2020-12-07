#include "../include/tensor.hpp"
#include <iostream>

int main()
{

    auto N = 5UL;
    for ( auto i = 1UL; i != N; ++i )
        for ( auto j = 1UL; j != N; ++j )
        {
            auto x = ceras::numeric::glorot_uniform<double>( {i, j} );
            std::cout << "glorot uniform random matrix for (" << i << "," << j << ")\n" << x << std::endl;
        }

    return 0;
}


