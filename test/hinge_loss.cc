#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include "../include/utils/color.hpp"
#include <cmath>
#include <iostream>

void test_1()
{
    std::cout << color::rize( "test_1", "Red" ) << std::endl;

    auto a = ceras::random<float>( {3, 3} );
    ceras::for_each( a.begin(), a.end(), []( auto& v ){ v = v > 0.5f ? 1.0 : -1.0; } );
    std::cout << "a created with:\n" << a << std::endl;
    auto b = ceras::random<float>( {3, 3} );
    ceras::for_each( b.begin(), b.end(), []( auto& v ){ v = v > 0.5f ? 1.0 : -1.0; } );
    std::cout << "b created with:\n" << b << std::endl;

    auto va = ceras::variable{ a };
    auto vb = ceras::variable{ b };
    auto diff = ceras::hinge_loss( va, vb );

    ceras::session<ceras::tensor<float>> s;
    auto d = s.run( diff );

    std::cout << "hinge loss is\n" << d << std::endl;


    std::cout << "axb=\n" << ceras::hadamard_product( a, b ) << std::endl;
}


int main()
{
    test_1();

    return 0;
}

