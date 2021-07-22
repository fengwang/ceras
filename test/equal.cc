#include "../include/ceras.hpp"
#include <iostream>

void test_43()
{
    auto a = ceras::linspace<double>( 1.0, 9.0, 9 );
    a.reshape( {3, 3} );
    std::cout << "a = \n" << a << std::endl;

    auto b = ceras::linspace<double>( 9.0, 1.0, 9 );
    b.reshape( {3, 3} );
    std::cout << "b = \n" << b << std::endl;

    auto va = ceras::variable{ a };
    auto vb = ceras::variable{ b };
    auto eq = ceras::equal( va, vb );

    auto& s = ceras::get_default_session<ceras::tensor<double>>();
    auto ans = s.run( eq );
    std::cout << "eq(a, b):\n" << ans << std::endl;

    auto grad = ceras::random<double>( {3, 3} );
    std::cout << "gradient generated as:\n" << grad << std::endl;
    eq.backward( grad );

    auto new_g = va.gradient();
    std::cout << "propageated gradient at a:\n" << new_g << std::endl;

    auto new_gb = vb.gradient();
    std::cout << "propageated gradient at b::\n" << new_gb << std::endl;
}

int main()
{
    test_43();

    return 0;
}

