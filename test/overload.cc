#include "../include/utils/overload.hpp"
#include <iostream>

int main()
{
    ceras::overload ov{
        []( int i ) { std::cout << "int: " << i << std::endl; },
        []( float f ) { std::cout << "float: " << f << std::endl; },
        []( double d ) { std::cout << "double: " << d << std::endl; }
    };

    ov( 1 );
    ov( 1.0f );
    ov( 1.0 );

    return 0;
}

