#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = ceras::variable{ ones<float>( {2, 2} ) };
    auto b = ceras::variable{ zeros<float>( {2, 2} ) };
    std::cout << "Testing minus with a=\n" << a.data() << std::endl;
    std::cout << "Testing minus with b=\n" << b.data() << std::endl;

    auto o = assign( a, b );

    auto& s = get_default_session<tensor<float>>();
    auto const& result = s.run( o );
    std::cout << "Result is " << result << std::endl;
    std::cout << "After assign a=\n" << a.data() << std::endl;
    std::cout << "After assign b=\n" << b.data() << std::endl;

    return 0;
}

