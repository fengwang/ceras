#include "../include/utils/range.hpp"

#include <iostream>

int main()
{
    {
        for ( auto idx : ceras::range( 10 ) )
            std::cout << idx << std::endl;
    }

    {
        for ( auto idx : ceras::range( 0, 10 ) )
            std::cout << idx << std::endl;
    }

    {
        for ( auto idx : ceras::range( 0, 10, 1 ) )
            std::cout << idx << std::endl;
    }

    return 0;
}

