#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include "../include/utils/color.hpp"
#include "../include/recurrent_operation.hpp"
#include <cmath>
#include <iostream>

void test_1()
{
    std::cout << color::rize( "test_1", "Red" ) << std::endl;
    auto a = ceras::linspace<double>( 1.0, 16.0, 16 );
    auto b = ceras::linspace<double>( -2.0, -17.0, 16 );
    a.reshape( {1, 4, 4, 1} );
    b.reshape( {1, 4, 4, 1} );
    std::cout << "a created with:\n" << ceras::squeeze(a) << std::endl;
    std::cout << "b created with:\n" << ceras::squeeze(b) << std::endl;

    auto va = ceras::variable{ a, true, true };
    auto vb = ceras::variable{ b, true, true };
    auto vab = va+vb;
    auto result = ceras::copy( vab, va );

    ceras::session<ceras::tensor<double>> s;
    auto ans = s.run( result );
    std::cout << "After session, the answer is:" << ceras::squeeze(ans) << std::endl;
    std::cout << "a is udpated to " << ceras::squeeze(a) << std::endl;

    result.reset_states();
    std::cout << "after resetting states, a is udpated to " << ceras::squeeze(a) << std::endl;
}

int main()
{
    test_1();

    return 0;
}

