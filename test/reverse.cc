#include "../include/utils/reverse.hpp"

#include <iostream>
#include <vector>

int main()
{
    std::vector<int> a{ {1, 2, 3, 5, 6, 7} };

    for ( auto x : ceras::reverse( a ) )
        std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}
