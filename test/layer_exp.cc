#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = variable{ ones<float>( {20,} ) };
    auto b = relu( Dense( 32, 20 )( a ) );
    auto la = exp(b);
    ceras::session<ceras::tensor<float>> s;
    std::cout <<  s.run( la );

    return 0;
}

