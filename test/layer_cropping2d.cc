#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    {
        auto a = ceras::variable{ random<float>( {1, 9, 9, 1}, -1.0f, 1.0f ) };
        std::cout << "Testing cropping_2d with a=\n" << squeeze(a.data()) << std::endl;

        auto la = ceras::cropping_2d({1,})(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << squeeze(result) << std::endl;

        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << squeeze(a.gradient()) << std::endl;
    }

    {
        auto a = ceras::variable{ random<float>( {1, 9, 9, 1}, -1.0f, 1.0f ) };
        std::cout << "Testing cropping_2d with a=\n" << squeeze(a.data()) << std::endl;

        auto la = ceras::cropping_2d({1,2})(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << squeeze(result) << std::endl;

        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << squeeze(a.gradient()) << std::endl;
    }

    {
        auto a = ceras::variable{ random<float>( {1, 9, 9, 1}, -1.0f, 1.0f ) };
        std::cout << "Testing cropping_2d with a=\n" << squeeze(a.data()) << std::endl;

        auto la = ceras::cropping_2d({1,2,3,4})(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << squeeze(result) << std::endl;

        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << squeeze(a.gradient()) << std::endl;
    }

    return 0;
}

