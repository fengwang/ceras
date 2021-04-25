#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = variable{ ones<float>( {2, 2} ) };
    auto ls = softmax(a);
    auto& s = ceras::get_default_session<ceras::tensor<float>>();
    std::cout <<  s.run( ls );

    return 0;
}

