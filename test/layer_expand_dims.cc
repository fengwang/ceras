#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = ceras::variable{ ceras::random<float>( {2, 3, 4}, -1.0f, 1.0f ) };
    auto& s = get_default_session<tensor<float>>();

    {
        auto la = ceras::expand_dims(0)(a);
        auto const& result = s.run( la );
        std::cout << "Resutl of expand_dims(0)( {2, 3, 4} ) is " << result << std::endl;
        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient is a=\n" << a.gradient() << std::endl;
    }

    {
        auto la = ceras::expand_dims(1)(a);
        auto const& result = s.run( la );
        std::cout << "Resutl of expand_dims(1)( {2, 3, 4} ) is " << result << std::endl;
        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient is a=\n" << a.gradient() << std::endl;
    }

    {
        auto la = ceras::expand_dims(2)(a);
        auto const& result = s.run( la );
        std::cout << "Resutl of expand_dims(2)( {2, 3, 4} ) is " << result << std::endl;
        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient is a=\n" << a.gradient() << std::endl;
    }

    {
        auto la = ceras::expand_dims(-1)(a);
        auto const& result = s.run( la );
        std::cout << "Resutl of expand_dims(-1)( {2, 3, 4} ) is " << result << std::endl;
        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient is a=\n" << a.gradient() << std::endl;
    }

    return 0;
}

