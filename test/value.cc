#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include "../include/utils/color.hpp"
#include <cmath>
#include <iostream>

void test_1()
{
    std::cout << color::rize( "test_1", "Red" ) << std::endl;

    auto a = ceras::linspace<float>( 1.0, 12.0, 12 );
    a.reshape( {3, 4} );
    std::cout << "a created with:\n" << a << std::endl;
    auto va = ceras::variable{ a };

    auto vb = ceras::value{ 1.1f };
    std::cout << "created value of  1.1f\n";
    ceras::session<ceras::tensor<float>> s;

    {
        auto v = va+vb;
        std::cout << "a+b = \n" << s.run( v );
    }
    {
        auto v = va-vb;
        std::cout << "a-b = \n" << s.run( v );
    }

    {
        auto v = hadamard_product( va, vb );
        std::cout << "a*b = \n" << s.run( v );
    }

}


int main()
{
    test_1();

    return 0;
}

