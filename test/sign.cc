#include "../include/ceras.hpp"
#include <iostream>

void test_43()
{
    auto a = ceras::linspace<double>( -9.0, 9.0, 9 );
    a.reshape( {3, 3} );
    std::cout << "a = \n" << a << std::endl;

    auto va = ceras::variable{ a };
    auto si = ceras::sign( va );

    auto& s = ceras::get_default_session<ceras::tensor<double>>();
    auto ans = s.run( si );
    std::cout << "sign(a):\n" << ans << std::endl;

    auto grad = ceras::random<double>( {3, 3} );
    std::cout << "gradient generated as:\n" << grad << std::endl;
    si.backward( grad );

    auto new_g = va.gradient();
    std::cout << "propageated gradient at a:\n" << new_g << std::endl;
}

int main()
{
    test_43();

    return 0;
}

