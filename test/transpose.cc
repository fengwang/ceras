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
    auto ta = ceras::transpose( va );

    ceras::session<double> s;
    auto ans = s.run( ta );
    std::cout << "after transpose:\n" << ans << std::endl;
}

int main()
{
    test_44();

    return 0;
}

