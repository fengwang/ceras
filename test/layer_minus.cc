#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = ceras::variable{ ones<float>( {2, 2} ) };
    auto b = ceras::variable{ ones<float>( {2, 2} ) };
    std::cout << "Testing minus with a=\n" << a.data() << std::endl;
    std::cout << "Testing minus with b=\n" << b.data() << std::endl;

    auto la = a - b;
    auto& s = get_default_session<tensor<float>>();
    auto const& result = s.run( la );
    std::cout << "Resutl is " << result << std::endl;

    la.backward( ceras::ones_like( result ) );
    std::cout << "gradient with a=\n" << a.gradient() << std::endl;
    std::cout << "gradient with b=\n" << b.gradient() << std::endl;

    return 0;
}

