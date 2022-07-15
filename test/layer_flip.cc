#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;


    {
        auto a = ceras::variable{ ceras::random<float>( {3, 5}, -1.0f, 1.0f ) };
        std::cout << "Testing flip(0) with a=\n" << a.data() << std::endl;

        auto la = ceras::flip( 0 )(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << result << std::endl;

        auto grad = ceras::random<float>( {3, 5}, -0.1f, 0.1f );
        std::cout << "backward gradient\n" << grad << std::endl;
        la.backward( grad );
        std::cout << "gradient with a=\n" << a.gradient() << std::endl;
    }

    {
        auto a = ceras::variable{ ceras::random<float>( {3, 5}, -1.0f, 1.0f ) };
        std::cout << "Testing flip(1) with a=\n" << a.data() << std::endl;

        auto la = ceras::flip( 1 )(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << result << std::endl;

        auto grad = ceras::random<float>( {3, 5}, -0.1f, 0.1f );
        std::cout << "backward gradient\n" << grad << std::endl;
        la.backward( grad );
        std::cout << "gradient with a=\n" << a.gradient() << std::endl;
    }

    return 0;
}

