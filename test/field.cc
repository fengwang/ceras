#include "../include/keras/field.hpp"

using namespace ceras::keras;


struct a : enabling_shape<a, 1, 1>, enabling_input_shape<a, 2, 2>, enabling_name<a, "...." >, enabling_alpha<a, "1.0" >
{};

#include <iostream>
int main()
{
    auto x = a().shape( {1, 2, 3, 4} ).input_shape( { 4, 5, 6} ).name( "nameofx" ).alpha( 3.14159265f );

    auto const& as = x.shape();
    {
        for ( auto v : as ) std::cout << v << " ";
        std::cout << std::endl;
    }

    auto const& ais = x.input_shape();
    {
        for ( auto v : ais ) std::cout << v << " ";
        std::cout << std::endl;
    }

    std::cout << "name: " <<  x.name() << std::endl;
    std::cout << "alpha: " << x.alpha() << std::endl;

    return 0;
}

