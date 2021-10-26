#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = ceras::variable{ ceras::random<float>( {3, 4}, -1.0f, 1.0f ) };
    auto& s = get_default_session<tensor<float>>();

    std::cout << "Got input tensor:\n" << a.data() << std::endl;

    {
        auto la = ceras::argmax(0)(a);
        auto const& result = s.run( la );
        std::cout << "Resutl of argmax(0)( {3, 4} ) is " << result << std::endl;
        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient is a=\n" << a.gradient() << std::endl;
    }

    {
        auto la = ceras::argmax(1)(a);
        auto const& result = s.run( la );
        std::cout << "Resutl of argmax(1)( { 3, 4} ) is " << result << std::endl;
        la.backward( ceras::ones_like( result ) );
        std::cout << "gradient is a=\n" << a.gradient() << std::endl;
    }


    return 0;
}

