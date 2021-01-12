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

    auto va = ceras::variable{ a };
    auto ta = ceras::max_pooling_2d( 2 )( va );

    ceras::session<ceras::tensor<double>> s;
    auto ans = s.run( ta );
    std::cout << "after max_pooling_2d(2):\n" << ceras::squeeze(ans) << std::endl;
}


void test_2()
{
    std::cout << color::rize( "test_2", "Red" ) << std::endl;
    auto a = ceras::linspace<double>( 1.0, 16.0, 16 );
    a.reshape( {1, 4, 4, 1} );
    std::cout << "a created with:\n" << ceras::squeeze(a) << std::endl;

    auto va = ceras::variable{ a };
    auto ta = ceras::max_pooling_2d( 2 )( va );

    ceras::session<ceras::tensor<double>> s;
    auto ans = s.run( ta );
    std::cout << "after max_pooling_2d(2):\n" << ceras::squeeze(ans) << std::endl;

    auto grad = ceras::tensor<double>( {1, 2, 2, 1}, {1.0, 2.0, 3.0, 4.0} );
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

