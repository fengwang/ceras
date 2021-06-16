#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = variable{ ones<float>( {20,} ) };
    auto b = relu( Dense( 32, 20 )( a ) );
    auto la = exponential(b);

    std::cout << computation_graph( la );

    return 0;
}

