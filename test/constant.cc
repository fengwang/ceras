#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include <cmath>
#include <iostream>

void test_45()
{
    auto a = ceras::linspace<double>( 1.0, 20.0, 20 );
    a.reshape( {4, 5} );
    std::cout << "a created with:\n" << a << std::endl;

    auto va = ceras::constant<double>{ a };
    auto ta = ceras::transpose( va );

    ceras::session<double> s;
    auto ans = s.run( ta );
    std::cout << "after transpose:\n" << ans << std::endl;
}

int main()
{
    test_45();

    return 0;
}

