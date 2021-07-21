#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = ceras::variable{ ceras::random<float>( {3, 5}, -1.0f, 1.0f )  };
    auto b = ceras::variable{ ceras::random<float>( {3, 5}, -1.0f, 1.0f )  };
    auto ab = ceras::concatenate(-1)( a, b );
    //auto ab = a+b;
    std::cout << "Testing relu6 with a=\n" << a.data() << std::endl;
    std::cout << "Testing relu6 with b=\n" << b.data() << std::endl;

    auto la = ceras::assign( a, ab );
    auto& s = get_default_session<tensor<float>>();
    auto const& result = s.run( la );
    std::cout << "a <- a+b: the resutl is " << result << std::endl;
    std::cout << "And updated a=\n" << a.data() << std::endl;

    la.backward( ceras::ones_like( result ) );
    std::cout << "gradient with a=\n" << a.gradient() << std::endl;

    return 0;
}

