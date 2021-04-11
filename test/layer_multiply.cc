#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = variable{ ones<float>( {2, 2} ) };
    auto b = variable{ ones<float>( {2, 2} ) };
    auto ab = a+b; // or 'auto ab = plus( a, b );'
    ceras::session<ceras::tensor<double>> s;
    std::cout <<  s.run( ab );

    return 0;
}

