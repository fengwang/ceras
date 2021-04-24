#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include "../include/utils/color.hpp"
#include <cmath>
#include <iostream>

void test_1()
{

    auto&& x = ceras::place_holder<ceras::tensor<float>>{};
    auto&& A = ceras::variable{ ceras::ones<float>({3, 3}) };
    auto&& b = ceras::variable{ ceras::zeros<float>({3,}) };
    auto&& z = sigmoid( A*x + b );

    auto&& X = ceras::tensor<float>{{3,}, {1.0f, 2.0f, 3.0f}};
    auto& s = ceras::get_default_session<ceras::tensor<float>>().get();
    s.bind(x, X);

    std::cout << "x=\n" << s.run(x) << std::endl;
    std::cout << "A=\n" << s.run(A) << std::endl;
    std::cout << "b=\n" << s.run(b) << std::endl;
    std::cout << "z=\n" << s.run(z) << std::endl;
}


int main()
{
    test_1();

    return 0;
}

