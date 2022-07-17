#include "../include/place_holder.hpp"
#include <iostream>

int main()
{
    {
        ceras::place_holder<ceras::tensor<float>> ph;
        auto const& [ph_name, ph_code] = ceras::serialize( ph );
        std::cout << "serialize place holder " << ph_name << "\n";
        std::cout << ph_code << "\n";
    }

    {
        ceras::place_holder<ceras::tensor<double>> ph;
        auto const& [ph_name, ph_code] = ceras::serialize( ph );
        std::cout << "serialize place holder " << ph_name << "\n";
        std::cout << ph_code << "\n";
    }

    return 0;
}

