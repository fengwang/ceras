#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = variable{ ones<float>( {2, 2} ) };
    auto ls = softmax(a);
    ceras::session<ceras::tensor<float>> s;
    std::cout <<  s.run( ls );

    return 0;
}

