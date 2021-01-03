#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include <cmath>
#include <iostream>

void test_44()
{
    auto a = ceras::linspace<double>( 1.0, 16.0, 16 );
    a.reshape( {4, 4} );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {1, 4, 4, 1} );

    auto b = ceras::tensor<double>{ {2, 2}, {-0.3, -0.1, 0.5, 0.7} };
    std::cout << "b created with:\n" << b << std::endl;
    b.reshape( {1, 2, 2, 1} );

    auto va = ceras::variable<double>{ a };
    auto vb = ceras::variable<double>{ b };
    auto cab = ceras::conv2d(4, 4)( va, vb );

    ceras::session<double> s;

    //auto sva = s.run( va );
    //std::cout << "va is\n" << sva << std::endl;


    auto ans = s.run( cab );
    std::cout << "after convolution:\n" << ceras::squeeze(ans) << std::endl;
}

int main()
{
    test_44();

    return 0;
}

