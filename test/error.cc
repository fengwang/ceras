#include "../include/ceras.hpp"

int main()
{
    auto a = ceras::linspace<double>( 1.0, 12.0, 12 );
    a.reshape( {1, 4, 3} );
    auto va = ceras::variable{ a };
    auto vb = ceras::variable{ a };
    auto vab = va+vb;
    int i = vab; // see error

    return 0;
}


