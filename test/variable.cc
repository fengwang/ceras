#include "../include/ceras.hpp"
#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include <cmath>
#include <iostream>

void test_serialize()
{
    auto a = ceras::linspace<double>( 1.0, 20.0, 20 );
    a.reshape( {4, 5} );
    std::cout << "a created with:\n" << a << std::endl;
    auto ca = ceras::variable<ceras::tensor<double>>{ a, 0.01, 0.02, false };
    auto caa = ceras::variable<ceras::tensor<double>>{ a, 0.021, 0.012, true };

    {
        auto const& [con_name, con_code] = ceras::serialize( ca );
        std::cout << "serialized variable " << con_name << "\n";
        std::cout << "serialized variable code:\n";
        for ( auto const& c : con_code )
            std::cout << c << std::endl;
    }

    {
        auto const& [con_name, con_code] = ceras::serialize( caa );
        std::cout << "serialized variable " << con_name << "\n";
        std::cout << "serialized variable code:\n";
        for ( auto const& c : con_code )
            std::cout << c << std::endl;
    }
}

int main()
{
    test_serialize();

    return 0;
}

