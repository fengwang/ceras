#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include "../include/utils/color.hpp"
#include <cmath>
#include <iostream>

void test_22()
{
    std::cout << color::rize( "test_22", "Red" ) << std::endl;
    auto a = ceras::linspace<double>( 0.0, 3.0, 4 );
    a.reshape( {2, 2} );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {1, 2, 2, 1} );

    auto b = ceras::tensor<double>{ {2, 2}, { 0.0, 1.0, 2.0, 3.0} };
    std::cout << "b created with:\n" << b << std::endl;
    b.reshape( {1, 2, 2, 1} );

    auto va = ceras::variable{ a };
    auto vb = ceras::variable{ b };
    auto cab = ceras::conv2d_transpose(2, 2)( va, vb );

    auto& s = ceras::get_default_session<ceras::tensor<double>>();

    auto ans = s.run( cab );
    std::cout << "after convolution:\n" << ceras::squeeze(ans) << std::endl;
}


int main()
{
    test_22();

    return 0;
}

