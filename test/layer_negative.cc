#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = ceras::variable{ ones<float>( {2, 2} ) };
    std::cout << "Testing negative with a=\n" << a.data() << std::endl;
    auto la = ceras::negative(a);
    auto& s = get_default_session<tensor<float>>();
    std::cout << "Resutl is " << s.run( la );

    return 0;
}

