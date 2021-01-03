#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include "../include/utils/color.hpp"
#include <cmath>
#include <iostream>

void test_44()
{
    std::cout << color::rize( "test_44", "Red" ) << std::endl;
    auto a = ceras::linspace<double>( 1.0, 16.0, 16 );
    a.reshape( {4, 4} );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {1, 4, 4, 1} );

    auto b = ceras::tensor<double>{ {2, 2}, {-0.3, -0.1, 0.5, 0.7} };
    std::cout << "b created with:\n" << b << std::endl;
    b.reshape( {1, 2, 2, 1} );

    auto va = ceras::variable<double>{ a };
    auto vb = ceras::variable<double>{ b };
    auto cab = ceras::conv2d(4, 4)( va, vb );

    ceras::session<double> s;

    auto ans = s.run( cab );
    std::cout << "after convolution:\n" << ceras::squeeze(ans) << std::endl;
}

void test_44_same()
{
    std::cout << color::rize( "test_44_same", "Red" ) << std::endl;

    auto a = ceras::linspace<double>( 1.0, 16.0, 16 );
    a.reshape( {4, 4} );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {1, 4, 4, 1} );

    auto b = ceras::tensor<double>{ {3, 3}, {1, 1, 1, 0, 0, 0, -1, -1, -1,} };

    std::cout << "b created with:\n" << b << std::endl;
    b.reshape( {1, 3, 3, 1} );

    auto va = ceras::variable<double>{ a };
    auto vb = ceras::variable<double>{ b };
    auto cab = ceras::conv2d(4, 4, 1, 1, 1, 1, "same")( va, vb );

    ceras::session<double> s;

    auto ans = s.run( cab );
    std::cout << "after convolution:\n" << ceras::squeeze(ans) << std::endl;
}

void test_44_22()
{
    std::cout << color::rize( "test_44_22", "Red" ) << std::endl;
    auto a = ceras::linspace<double>( 1.0, 16.0, 16 );
    a.reshape( {4, 4} );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {1, 4, 4, 1} );

    auto b = ceras::tensor<double>{ {2, 2, 2}, {1, 2, 3, 4, -4, -3, -2, -1} };
    std::cout << "b created with:\n" << b << std::endl;
    b.reshape( {2, 2, 2, 1} );

    auto va = ceras::variable<double>{ a };
    auto vb = ceras::variable<double>{ b };
    auto cab = ceras::conv2d(4, 4)( va, vb );

    ceras::session<double> s;

    auto ans = s.run( cab );
    std::cout << "after convolution:\n" << ceras::squeeze(ans) << std::endl;
}

void test_55_33()
{
    std::cout << color::rize( "test_55_33", "Red" ) << std::endl;
    auto a = ceras::linspace<double>( 1.0, 25.0, 25 );
    a.reshape( {5, 5} );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {1, 5, 5, 1} );

    auto b = ceras::tensor<double>{ {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9} };
    std::cout << "b created with:\n" << b << std::endl;
    b.reshape( {1, 3, 3, 1} );

    auto va = ceras::variable<double>{ a };
    auto vb = ceras::variable<double>{ b };
    auto cab = ceras::conv2d(5, 5)( va, vb );

    ceras::session<double> s;

    auto ans = s.run( cab );
    std::cout << "after convolution:\n" << ceras::squeeze(ans) << std::endl;
}

void test_55_33_same()
{
    std::cout << color::rize( "test_55_33_valid", "Red" ) << std::endl;
    auto a = ceras::linspace<double>( 1.0, 25.0, 25 );
    a.reshape( {5, 5} );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {1, 5, 5, 1} );

    auto b = ceras::tensor<double>{ {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9} };
    std::cout << "b created with:\n" << b << std::endl;
    b.reshape( {1, 3, 3, 1} );

    auto va = ceras::variable<double>{ a };
    auto vb = ceras::variable<double>{ b };
    auto cab = ceras::conv2d(5, 5, 1, 1, 1, 1, "same")( va, vb );

    ceras::session<double> s;

    auto ans = s.run( cab );
    std::cout << "after convolution:\n" << ceras::squeeze(ans) << std::endl;
}

int main()
{
    test_44();

    test_44_same();

    test_44_22();

    test_55_33();

    test_55_33_same();

    return 0;
}

