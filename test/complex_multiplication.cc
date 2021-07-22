#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    {
        auto a = ceras::variable{ ones<float>( {2, 3} ) };
        auto b = ceras::variable{ ones<float>( {2, 3} ) };
        auto c = ceras::complex{ a, b };
        auto d = ceras::variable{ ones<float>( {3, 2} ) };
        auto e = ceras::variable{ ones<float>( {3, 2} ) };
        auto f = ceras::complex{ d, e };
        std::cout << "Testing complex with a=\n" << a.data() << std::endl;
        std::cout << "Testing complex with b=\n" << b.data() << std::endl;
        std::cout << "Testing complex with d=\n" << d.data() << std::endl;
        std::cout << "Testing complex with e=\n" << e.data() << std::endl;

        auto la = ceras::abs( c * f );
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << result << std::endl;

        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << a.gradient() << std::endl;
        std::cout << "gradient with b=\n" << b.gradient() << std::endl;
        std::cout << "gradient with d=\n" << d.gradient() << std::endl;
        std::cout << "gradient with e=\n" << e.gradient() << std::endl;
    }

    return 0;
}

