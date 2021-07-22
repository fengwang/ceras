#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = variable{ ones<float>( {2, 2} ) };
    auto b = variable{ zeros<float>( {2, 2} ) };
    auto ab = a+b; // or 'auto ab = plus( a, b );'
    auto& s = ceras::get_default_session<ceras::tensor<float>>();
    std::cout <<  s.run( ab );

    return 0;
}

