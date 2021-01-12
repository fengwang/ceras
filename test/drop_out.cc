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
    auto ta = ceras::drop_out( 0.2 )( va );

    ceras::session<ceras::tensor<double>> s;
    auto ans = s.run( ta );
    std::cout << "after drop_out (0.2):\n" << ans << std::endl;

    auto ra = ceras::drop_out( 0.4 )( va );
    auto rns = s.run( ra );
    std::cout << "after drop_out (0.4):\n" << rns << std::endl;
}


void test_2()
{
    std::cout << color::rize( "test_2", "Red" ) << std::endl;

    auto a = ceras::linspace<double>( 1.0, 12.0, 12 );
    a.reshape( {3, 4} );
    std::cout << "a created with:\n" << a << std::endl;

    auto va = ceras::variable{ a };
    auto ta = ceras::drop_out( 0.5 )( va );

    ceras::session<ceras::tensor<double>> s;
    auto ans = s.run( ta );
    std::cout << "after drop_out (0.5):\n" << ans << std::endl;


    auto grad = ceras::ones<double>( {3, 4} );
    std::cout << "gradient generated as:\n" << grad << std::endl;
    ta.backward( grad );

    auto new_g = *(va.gradient_);
    std::cout << "propageated gradient:\n" << new_g << std::endl;
}



void test_3()
{
    std::cout << color::rize( "test_3: in case not learning", "Red" ) << std::endl;

    auto a = ceras::linspace<double>( 1.0, 12.0, 12 );
    a.reshape( {3, 4} );
    std::cout << "a created with:\n" << a << std::endl;

    auto va = ceras::variable{ a };
    auto ta = ceras::drop_out( 0.5 )( va );

    ceras::session<ceras::tensor<double>> s;
    ceras::learning_phase = 0;
    auto ans = s.run( ta );
    std::cout << "after drop_out (0.5):\n" << ans << std::endl;


    auto grad = ceras::ones<double>( {3, 4} );
    std::cout << "gradient generated as:\n" << grad << std::endl;
    ta.backward( grad );

    auto new_g = *(va.gradient_);
    std::cout << "propageated gradient:\n" << new_g << std::endl;
}


int main()
{
    test_1();
    test_2();
    test_3();

    return 0;
}

