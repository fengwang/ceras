#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include "../include/utils/color.hpp"
#include <cmath>
#include <iostream>

void test_1()
{
    std::cout << color::rize( "test_1", "Red" ) << std::endl;
    auto a = ceras::linspace<double>( 1.0, 12.0, 12 );
    a.reshape( {3, 4} );
    std::cout << "a created with:\n" << a << std::endl;

    auto va = ceras::variable<double>{ a };
    auto ta = ceras::reshape({2, 6}, true)( va );

    ceras::session<double> s;
    auto ans = s.run( ta );
    std::cout << "after transpose with batch size:\n" << ans << std::endl;

    auto ra = ceras::reshape({2, 6}, false)( va );
    auto rns = s.run( ra );
    std::cout << "after transpose without batch size:\n" << rns << std::endl;
}


void test_2()
{
    std::cout << color::rize( "test_2", "Red" ) << std::endl;
    auto a = ceras::linspace<double>( 1.0, 12.0, 12 );
    a.reshape( {3, 4} );

    auto va = ceras::variable<double>{ a };
    auto ta = ceras::reshape({2, 6})( va );

    ceras::session<double> s;
    auto ans = s.run( ta );

    auto grad = ceras::random<double>( {1, 2, 6} );
    std::cout << "gradient generated as:\n" << grad << std::endl;
    ta.backward( grad );

    auto new_g = *(va.gradient_);
    std::cout << "propageated gradient:\n" << new_g << std::endl;
}




int main()
{
    test_1();
    test_2();

    return 0;
}

