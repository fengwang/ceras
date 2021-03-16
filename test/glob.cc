#include "../include/utils/3rd_party/glob.hpp"
#include <iostream>

int main()
{
    auto paths = glob::glob("./test/*.cc");
    std::cout << "There are " << paths.size() << " cpp source files under test folder:\n";
    for ( auto path : paths )
        std::cout << path.string() << std::endl;

    return 0;
}


