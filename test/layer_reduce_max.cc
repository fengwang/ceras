#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    {
        auto a = ceras::variable{ random<float>( {2, 3}, -1.0f, 1.0f ) };
        std::cout << "Testing reduce_max(0) with a=\n" << a.data() << std::endl;

        auto la = ceras::reduce_max(0)(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << result << std::endl;

        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << a.gradient() << std::endl;
    }

    {
        auto a = ceras::variable{ random<float>( {3, 4}, -1.0f, 1.0f ) };
        std::cout << "Testing reduce_max(1) with a=\n" << a.data() << std::endl;

        auto la = ceras::reduce_max(1)(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << result << std::endl;

        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << a.gradient() << std::endl;
    }

    {
        auto a = ceras::variable{ random<float>( {4, 5}, -1.0f, 1.0f ) };
        std::cout << "Testing reduce_max() with a=\n" << a.data() << std::endl;

        auto la = ceras::reduce_max()(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << result << std::endl;

        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << a.gradient() << std::endl;
    }

    return 0;
}

