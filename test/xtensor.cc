#include "../include/utils/3rd_party/xtensor.hpp"
#include "../include/utils/fmt.hpp"
#include <iostream>

struct dummy{};

void test_1()
{
    xt::xarray<float> xf;
    std::vector<unsigned long> arr;
    {
        arr.resize( 4 );
        std::fill( arr.begin(), arr.end(), 10 );
    }

    xf.resize( arr );

    std::cout << xf( 1, 2, 3, 4 ) << std::endl;

    xf(0, 0, 0, 0 ) = 1.2;
    std::cout << xf( 0, 0, 0, 0 ) << std::endl;

    xf(0) = 2.2;
    std::cout << xf( 0, 0, 0, 0 ) << std::endl;


    float* v = &xf(0);
    *v = 3.3;
    std::cout << xf( 0, 0, 0, 0 ) << std::endl;

    xf.resize( {100, 1, 10, 10 } );
    std::cout << xf( 0, 0, 0, 0 ) << std::endl;

    std::cout << xf.size() << std::endl;

    std::cout << fmt::format( "got shape: {}", xf.shape() ) << std::endl;

}


void test_2()
{
    std::vector<unsigned long> arr;
    {
        arr.resize( 4 );
        std::fill( arr.begin(), arr.end(), 10 );
    }
    xt::xarray<float> xf;
    xf.resize( arr );

    std::fill(xf.begin(), xf.end(), 1.1);
    std::cout << xf( 1, 2, 3, 4 ) << std::endl;


    std::for_each( xf.data(), xf.data()+xf.size(), []( auto& x ) { x -= 1.1; } );
    std::cout << xf( 4, 2, 3, 4 ) << std::endl;
}

void test_3()
{
    std::vector<unsigned long> arr;
    {
        arr.resize( 4 );
        std::fill( arr.begin(), arr.end(), 10 );
    }
    xt::xarray<float> xf;
    xf.resize( arr );

    std::cout << "data position: " << xf.data() << " -- " << xf.begin() << " -- " << xf.end() << std::endl;

    //dummy d = xf.data();
    //dummy e = xf.begin();
}

void test_4()
{
    std::vector<unsigned long> arr;
    {
        arr.resize( 4 );
        std::fill( arr.begin(), arr.end(), 10 );
    }
    xt::xarray<float> xf;
    xf.resize( arr );

    float v = 0.0;
    std::for_each( xf.data(), xf.data()+xf.size(), [&v]( auto& x ) { x += v; v+=0.1; } );


    //auto xfa = xt::sum( xf, {1,2},  xt::keep_dims | xt::evaluation_strategy::immediate );
    auto xfa = xt::sum( xf, std::vector<unsigned long>{{1,2}},  xt::keep_dims | xt::evaluation_strategy::immediate );
    std::cout << xfa.size() << " : " << xfa( 0, 0, 0, 1 ) << std::endl;

}


int main()
{
    //test_1();
    //test_2();
    //test_3();
    test_4();

    return 0;
}



