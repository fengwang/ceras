#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = ceras::variable{ ceras::random<float>( {3, 2}, -1.0f, 1.0f ) };

    {
        float exp = 1.0;
        std::cout << "Testing pow with a=\n" << a.data() << "and exp = " << exp << std::endl;
        auto la = ceras::pow(a, exp);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << result << std::endl;
        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << a.gradient() << std::endl;
    }

    {
        float exp = 0.0;
        std::cout << "Testing pow with a=\n" << a.data() << "and exp = " << exp << std::endl;
        auto la = ceras::pow(a, exp);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << result << std::endl;
        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << a.gradient() << std::endl;
    }

    {
        float exp = -1.0;
        std::cout << "Testing pow with a=\n" << a.data() << "and exp = " << exp << std::endl;
        auto la = ceras::pow(a, exp);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << result << std::endl;
        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << a.gradient() << std::endl;
    }

    return 0;
}

