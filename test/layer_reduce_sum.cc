#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    {
        auto a = ceras::variable{ ones<float>( {2, 3} ) };
        std::cout << "Testing reduce_sum(0) with a=\n" << a.data() << std::endl;

        auto la = ceras::reduce_sum(0)(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << result << std::endl;

        la.backward( ceras::random_like( result ) );
        std::cout << "gradient with a=\n" << a.gradient() << std::endl;
    }

    {
        auto a = ceras::variable{ ones<float>( {3, 4} ) };
        std::cout << "Testing reduce_sum(1) with a=\n" << a.data() << std::endl;

        auto la = ceras::reduce_sum(1)(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << result << std::endl;

        la.backward( ceras::random_like( result ) );
        std::cout << "gradient with a=\n" << a.gradient() << std::endl;
    }

    {
        auto a = ceras::variable{ ones<float>( {4, 5} ) };
        std::cout << "Testing reduce_sum(-1) with a=\n" << a.data() << std::endl;

        auto la = ceras::reduce_sum(-1)(a);
        auto& s = get_default_session<tensor<float>>();
        auto const& result = s.run( la );
        std::cout << "Resutl is " << result << std::endl;

        la.backward( ceras::random_like( result ) );
        std::cout << "gradient with a=\n" << a.gradient() << std::endl;
    }

    return 0;
}

