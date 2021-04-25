#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include "../include/utils/color.hpp"
#include <cmath>
#include <iostream>

using namespace ceras;

void test_1()
{
    std::cout << color::rize( "test_1", "Red" ) << std::endl;

    auto vx = variable{ randn<double>( {128, 128} ) };
    auto nx = normalization()( vx );

    auto& s = ceras::get_default_session<ceras::tensor<double>>();
    auto ans = s.run( nx );

    auto m_ans = mean( ans, 0 );
    auto v_ans = variance( ans, 0 );

    std::cout << "normalization, mean of ans is :\n" << m_ans << std::endl;
    std::cout << "normalization, variance of ans is :\n" << v_ans << std::endl;
}

int main()
{
    test_1();

    return 0;
}

