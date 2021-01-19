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

    auto va = ceras::variable{ a };
    auto vb = ceras::variable{ b };
    auto cab = ceras::conv2d(4, 4)( va, vb );

    ceras::session<ceras::tensor<double>> s;

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

    auto va = ceras::variable{ a };
    auto vb = ceras::variable{ b };
    auto cab = ceras::conv2d(4, 4, 1, 1, 1, 1, "same")( va, vb );

    ceras::session<ceras::tensor<double>> s;

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

    auto va = ceras::variable{ a };
    auto vb = ceras::variable{ b };
    auto cab = ceras::conv2d(4, 4)( va, vb );

    ceras::session<ceras::tensor<double>> s;

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

    auto va = ceras::variable{ a };
    auto vb = ceras::variable{ b };
    auto cab = ceras::conv2d(5, 5)( va, vb );

    ceras::session<ceras::tensor<double>> s;

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

    auto va = ceras::variable{ a };
    auto vb = ceras::variable{ b };
    auto cab = ceras::conv2d(5, 5, 1, 1, 1, 1, "same")( va, vb );

    ceras::session<ceras::tensor<double>> s;

    auto ans = s.run( cab );
    std::cout << "after convolution:\n" << ceras::squeeze(ans) << std::endl;
}

void test_55_33_same_back()
{
    std::cout << color::rize( "test_55_33_valid", "Red" ) << std::endl;
    auto a = ceras::linspace<double>( 1.0, 25.0, 25 );
    a.reshape( {5, 5} );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {1, 5, 5, 1} );

    //auto b = ceras::tensor<double>{ {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9} };
    auto b = ceras::ones<double>( {3, 3} );
    std::cout << "b created with:\n" << b << std::endl;
    b.reshape( {1, 3, 3, 1} );

    auto va = ceras::variable{ a };
    auto vb = ceras::variable{ b };
    auto cab = ceras::conv2d(5, 5, 1, 1, 1, 1, "same")( va, vb );

    ceras::session<ceras::tensor<double>> s;

    auto ans = s.run( cab );
    std::cout << "after convolution:\n" << ceras::squeeze(ans) << std::endl;

    auto grad = ceras::ones<double>( {1, 5, 5, 1} );
    cab.backward( grad );

    std::cout << "after backward, gradient for a is updated to :\n" << ceras::squeeze(*(va.gradient_)) << std::endl;
}

void test_55_33_same_back_s2()
{
    std::cout << color::rize( "test_55_33_valid", "Red" ) << std::endl;
    auto a = ceras::linspace<double>( 1.0, 25.0, 25 );
    a.reshape( {5, 5} );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {1, 5, 5, 1} );

    auto b = ceras::ones<double>( {3, 3} );
    std::cout << "b created with:\n" << b << std::endl;
    b.reshape( {1, 3, 3, 1} );

    auto va = ceras::variable{ a };
    auto vb = ceras::variable{ b };
    auto cab = ceras::conv2d(5, 5, 1, 1, 1, 1, "same")( va, vb );

    ceras::session<ceras::tensor<double>> s;

    auto ans = s.run( cab );
    std::cout << "after convolution:\n" << ceras::squeeze(ans) << std::endl;

    auto grad = ceras::ones<double>( {1, 5, 5, 1} );
    cab.backward( grad );

    std::cout << "after backward, gradient for a is updated to :\n" << ceras::squeeze(*(va.gradient_)) << std::endl;
}

void test_66_33_same()
{
    std::cout << color::rize( "test_66_33_same", "Red" ) << std::endl;
    auto a = ceras::linspace<double>( 1.0, 36.0, 36 );
    a.reshape( {6, 6} );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {1, 6, 6, 1} );

    auto b = ceras::ones<double>( {3, 3} );
    std::cout << "b created with:\n" << b << std::endl;
    b.reshape( {1, 3, 3, 1} );

    auto va = ceras::variable{ a };
    auto vb = ceras::variable{ b };
    auto cab = ceras::conv2d(6, 6, 3, 3, 1, 1, "same")( va, vb );

    ceras::session<ceras::tensor<double>> s;

    auto ans = s.run( cab );
    std::cout << "after convolution:\n" << ceras::squeeze(ans) << std::endl;

    auto grad = ceras::ones<double>( {1, 2, 2, 1} );
    cab.backward( grad );

    std::cout << "after backward, gradient for a is updated to :\n" << ceras::squeeze(*(va.gradient_)) << std::endl;
}

int main()
{
#if 1
    test_44();

    test_44_same();

    test_44_22();

    test_55_33();

    test_55_33_same();

    test_55_33_same_back();

    test_55_33_same_back_s2();
#endif
    test_66_33_same();

    return 0;
}

