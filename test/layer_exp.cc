#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = variable{ ones<float>( {20,} ) };
    auto b = relu( Dense( 32, 20 )( a ) );
    auto la = exponential(b);
    //auto& s = ceras::get_default_session<ceras::tensor<float>>();
    auto& s = ceras::get_default_session<ceras::tensor<float>>();
    std::cout <<  s.run( la );

    return 0;
}

