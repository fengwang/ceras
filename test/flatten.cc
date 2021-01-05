#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include <cmath>
#include <iostream>

void test_44()
{
    auto a = ceras::linspace<double>( 1.0, 16.0, 16 );
    a.reshape( {4, 4} );
    std::cout << "a created with:\n" << a << std::endl;

    auto va = ceras::variable<double>{ a };
    auto ta = ceras::flatten( va );

    ceras::session<double> s;
    auto ans = s.run( ta );
    std::cout << "(4, 4) after flatten:\n" << ans << std::endl;
}


void test_44_back()
{
    auto a = ceras::linspace<double>( 1.0, 12.0, 12 );
    a.reshape( {1, 4, 3} );

    auto va = ceras::variable<double>{ a };
    auto ta = ceras::flatten( va );

    ceras::session<double> s;
    auto ans = s.run( ta );
    std::cout << "(1, 4, 3) after flatten:\n" << ans << std::endl;

    auto grad = ceras::random<double>( {1, 12} );
    std::cout << "gradient generated as:\n" << grad << std::endl;
    ta.backward( grad );

    auto new_g = *(va.gradient_);
    std::cout << "propageated gradient:\n" << new_g << std::endl;
}




int main()
{
    test_44();

    test_44_back();

    return 0;
}
