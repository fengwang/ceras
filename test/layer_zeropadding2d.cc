#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    {
        auto a = ceras::variable{ ones<float>( {1, 2, 3, 1}, -1.0f, 1.0f ) };
        std::cout << "Testing zero_padding_2d with a=\n" << a.data() << std::endl;

        auto la = ceras::zero_padding_2d({1,})(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << result << std::endl;

        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << a.gradient() << std::endl;
    }

    {
        auto a = ceras::variable{ ones<float>( {1, 2, 3, 1}, -1.0f, 1.0f ) };
        std::cout << "Testing zero_padding_2d with a=\n" << a.data() << std::endl;

        auto la = ceras::zero_padding_2d({1,2})(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << result << std::endl;

        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << a.gradient() << std::endl;
    }

    {
        auto a = ceras::variable{ ones<float>( {1, 2, 3, 1}, -1.0f, 1.0f ) };
        std::cout << "Testing zero_padding_2d with a=\n" << a.data() << std::endl;

        auto la = ceras::zero_padding_2d({1,2,3,4})(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << result << std::endl;

        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient with a=\n" << a.gradient() << std::endl;
    }

    return 0;
}

