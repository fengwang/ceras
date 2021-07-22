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

    auto b = ceras::linspace<double>( 1.0, 12.0, 12 );
    b.reshape( {3, 4} );
    std::cout << "b created with shape (3, 4) in range [1, 12]:\n";

    auto va = ceras::variable{ a };
    auto vb = ceras::variable{ b };
    auto ta = ceras::elementwise_product( va, vb );

    //auto& s = ceras::get_default_session<ceras::tensor<double>>();
    auto& s = ceras::get_default_session<ceras::tensor<double>>();
    auto ans = s.run( ta );

    std::cout << "after elementwise_product(va, vb):\n" << ans << std::endl;
}

void test_2()
{
    std::cout << color::rize( "test_2", "Green" ) << std::endl;
    auto a = ceras::linspace<double>( 1.0, 6.0, 6 );
    a.reshape( {3, 2} );
    std::cout << "a created with:\n" << a << std::endl;

    auto b = ceras::linspace<double>( 1.0, 12.0, 12 );
    b.reshape( {2, 3, 2} );
    std::cout << "b created with shape (2, 3, 2) in range [1, 12]:\n";

    auto va = ceras::variable{ a };
    auto vb = ceras::variable{ b };
    auto ta = ceras::elementwise_product( va, vb );

    auto& s = ceras::get_default_session<ceras::tensor<double>>();
    auto ans = s.run( ta );
    std::cout << "after elementwise_product(va, vb):\n" << ans << std::endl;

    ta.backward( ceras::ones<double>({2, 3, 2}) );

    std::cout << "after backward with ones, the va gradient is:\n" << va.gradient() << std::endl;
    std::cout << "after backward with ones, the vb gradient is:\n" << vb.gradient() << std::endl;

}

int main()
{
    test_1();
    test_2();

    return 0;
}

