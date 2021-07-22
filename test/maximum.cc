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
    auto va = ceras::variable{ a };


    auto b = ceras::linspace<double>( 12.0, 1.0, 12 );
    b.reshape( {3, 4} );
    std::cout << "b created with:\n" << b << std::endl;
    auto vb = ceras::variable{ b };

    auto mab = maximum( va, vb );

    auto& s = ceras::get_default_session<ceras::tensor<double>>();
    auto ans = s.run( mab );
    std::cout << "after maximum(a, b):\n" << ans << std::endl;


    auto grad = ceras::ones<double>( {3, 4} );
    std::cout << "gradient generated as:\n" << grad << std::endl;
    mab.backward( grad );

    auto a_g = va.gradient();
    std::cout << "propageated gradient at a:\n" << a_g << std::endl;

    auto b_g = vb.gradient();
    std::cout << "propageated gradient at b:\n" << b_g << std::endl;
}


int main()
{
    test_1();

    return 0;
}

