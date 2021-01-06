#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include "../include/utils/color.hpp"
#include <cmath>
#include <iostream>

void test_1()
{
    std::cout << color::rize( "test_1", "Red" ) << std::endl;
    auto a = ceras::linspace<double>( 1.0, 16.0, 16 );
    a.reshape( {1, 4, 4, 1} );
    std::cout << "a created with:\n" << ceras::squeeze(a) << std::endl;

    auto va = ceras::variable<double>{ a };
    auto ta = ceras::up_sampling_2d( 2 )( va );

    ceras::session<double> s;
    auto ans = s.run( ta );
    std::cout << "after up_sampling_2d(2):\n" << ceras::squeeze(ans) << std::endl;
}


void test_2()
{
    std::cout << color::rize( "test_2", "Red" ) << std::endl;
    auto a = ceras::linspace<double>( 1.0, 16.0, 16 );
    a.reshape( {1, 4, 4, 1} );
    std::cout << "a created with:\n" << ceras::squeeze(a) << std::endl;

    auto va = ceras::variable<double>{ a };
    auto ta = ceras::up_sampling_2d( 2 )( va );

    ceras::session<double> s;
    auto ans = s.run( ta );
    std::cout << "after up_sampling_2d(2):\n" << ceras::squeeze(ans) << std::endl;

    auto grad = ceras::linspace<double>( 1.0, 64.0, 64 );
    grad.reshape( {1, 8, 8, 1} );
    std::cout << "gradient generated as:\n" << ceras::squeeze(grad) << std::endl;
    ta.backward( grad );

    auto new_g = *(va.gradient_);
    std::cout << "propageated gradient:\n" << ceras::squeeze(new_g) << std::endl;
}


int main()
{
    test_1();
    test_2();

    return 0;
}

