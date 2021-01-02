#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include <cmath>
#include <iostream>

/*
void img2col_test( std::size_t rows, std::size_t cols, std::size_t chs, std::size_t kr, std::size_t kc, std::size_t pr, std::size_t pc, std::size_t sr, std::size_t sc, std::size_t dr, std::size_t dc )
{
}
*/


void test_55_33()
{
    auto a = ceras::linspace<double>( 1.0, 32.0, 32 );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {1, 4, 4, 2} );

    auto va = ceras::variable<double>{ a };
    auto col_va = ceras::img2col(3,3)( va );

    ceras::session<double> s;
    auto col_im = s.run( col_va );
    std::cout << "after col2img:\n" << col_im << std::endl;

    {
        auto col_im = s.run( col_va );
        std::cout << "(2nd round) after col2img:\n" << col_im << std::endl;
    }
}

void test_332_33()
{
    auto a = ceras::linspace<double>( 1.0, 18.0, 18 );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {1, 3, 3, 2} );

    auto va = ceras::variable<double>{ a };
    auto col_va = ceras::img2col(2,2)( va );

    ceras::session<double> s;
    auto col_im = s.run( col_va );
    std::cout << "after col2img:\n" << col_im << std::endl;
}

void test_3320_33()
{
    auto a = ceras::linspace<double>( 1.0, 18.0, 18 );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {1, 3, 3, 2} );

    auto va = ceras::variable<double>{ a };
    auto col_va = ceras::img2col(2, 2, 1, 1)( va );

    ceras::session<double> s;
    auto col_im = s.run( col_va );
    std::cout << "after col2img:\n" << col_im << std::endl;
}

void test_2332_33()
{
    auto a = ceras::linspace<double>( 1.0, 36.0, 36 );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {2, 3, 3, 2} );

    auto va = ceras::variable<double>{ a };
    auto col_va = ceras::img2col(2,2)( va );

    ceras::session<double> s;
    auto col_im = s.run( col_va );
    std::cout << "after col2img:\n" << col_im << std::endl;
}

void test_3332_33()
{
    auto a = ceras::linspace<double>( 1.0, 54.0, 54 );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {3, 3, 3, 2} );

    auto va = ceras::variable<double>{ a };
    auto col_va = ceras::img2col(2,2)( va );

    ceras::session<double> s;
    auto col_im = s.run( col_va );
    std::cout << "after col2img:\n" << col_im << std::endl;
}

void test_1441_22_s2()
{
    auto a = ceras::linspace<double>( 1.0, 16.0, 16 );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {1, 4, 4, 1} );

    auto va = ceras::variable<double>{ a };
    auto col_va = ceras::img2col(2, 2, 0, 0, 2, 2)( va );

    ceras::session<double> s;
    auto col_im = s.run( col_va );
    std::cout << "after col2img:\n" << col_im << std::endl;
}

void test_1442_22_s2()
{
    auto a = ceras::linspace<double>( 1.0, 32.0, 32 );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {1, 4, 4, 2} );

    auto va = ceras::variable<double>{ a };
    auto col_va = ceras::img2col(2, 2, 0, 0, 2, 2)( va );

    ceras::session<double> s;
    auto col_im = s.run( col_va );
    std::cout << "after col2img:\n" << col_im << std::endl;
}

void test_2442_22_s2()
{
    auto a = ceras::linspace<double>( 1.0, 32.0*2, 32*2 );
    std::cout << "a created with:\n" << a << std::endl;
    a.reshape( {2, 4, 4, 2} );

    auto va = ceras::variable<double>{ a };
    auto col_va = ceras::img2col(2, 2, 0, 0, 2, 2)( va );

    ceras::session<double> s;
    auto col_im = s.run( col_va );
    std::cout << "after col2img:\n" << col_im << std::endl;
}


int main()
{
    using namespace ceras;

#if 0
    test_55_33();
    test_3320_33();

    std::cout << "<--------------->" << std::endl;
    test_332_33();

    std::cout << "---------------" << std::endl;
    test_3332_33();

    std::cout << ">---------------<" << std::endl;
    test_2332_33();
#endif

    std::cout << "Test Stride" << std::endl;
    test_1441_22_s2();

    test_1442_22_s2();

    test_2442_22_s2();

    return 0;
}

