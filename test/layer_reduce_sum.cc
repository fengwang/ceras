#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = ceras::variable{ ones<float>( {2, 2} ) };
    std::cout << "Testing reduce_sum with a=\n" << a.data() << std::endl;

    auto la = ceras::reduce_sum(a);
    auto& s = get_default_session<tensor<float>>();
    auto const& result = s.run( la );
    std::cout << "Resutl is " << result << std::endl;

    la.backward( ceras::ones_like( result ) );
    std::cout << "gradient with a=\n" << a.gradient() << std::endl;

    return 0;
}

