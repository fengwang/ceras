#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = variable{ ones<float>( {2, 2} ) };
    auto la = log(a);
    //auto& s = ceras::get_default_session<ceras::tensor<float>>();
    auto& s = ceras::get_default_session<ceras::tensor<float>>();
    std::cout <<  s.run( la );

    return 0;
}

