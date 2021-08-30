#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    {
        auto a = ceras::variable{ ones<float>( {1, 6, 7, 1} ) };
        std::cout << "Testing sliding_2d with a=\n" << squeeze(a.data()) << std::endl;

        auto la = ceras::sliding_2d({3,})(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << squeeze(result) << std::endl;

        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << squeeze(a.gradient()) << std::endl;
    }


    {
        auto a = ceras::variable{ ones<float>( {1, 6, 7, 1} ) };
        std::cout << "Testing sliding_2d with a=\n" << squeeze(a.data()) << std::endl;

        auto la = ceras::sliding_2d({3,})(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << squeeze(result) << std::endl;

        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << squeeze(a.gradient()) << std::endl;
    }

    {
        auto a = ceras::variable{ ones<float>( {1, 6, 6, 1} ) };
        std::cout << "Testing sliding_2d with a=\n" << squeeze(a.data()) << std::endl;

        auto la = ceras::sliding_2d({2})(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << squeeze(result) << std::endl;

        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << squeeze(a.gradient()) << std::endl;
    }

    {
        auto a = ceras::variable{ ones<float>( {1, 5, 7, 1} ) };
        std::cout << "Testing sliding_2d with a=\n" << squeeze(a.data()) << std::endl;

        auto la = ceras::sliding_2d({3})(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << squeeze(result) << std::endl;

        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << squeeze(a.gradient()) << std::endl;
    }

    return 0;
}

