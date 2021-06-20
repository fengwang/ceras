#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = variable{ ones<float>( {2, 2} ) };
    std::cout << "Testing log with a=\n" << a.data() << std::endl;
    auto la = log(a);
    auto& s = get_default_session<tensor<float>>();
    std::cout <<  s.run( la );

    return 0;
}

