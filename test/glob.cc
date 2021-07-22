#include "../include/utils/3rd_party/glob.hpp"
#include <iostream>

int main()
{
    auto paths = glob::glob("./test/*.cc");
    std::cout << "There are " << paths.size() << " cpp source files under test folder:\n";
    for ( auto const& path : paths )
        std::cout << path.string() << std::endl;

    {
        auto paths = glob::glob("./examples/vgg16/dogs-vs-cats/train/cat.*.jpg");
        std::cout << "There are " << paths.size() << " cat images\n";
    }
    {
        auto paths = glob::glob("./examples/vgg16/dogs-vs-cats/train/dog.*.jpg");
        std::cout << "There are " << paths.size() << " dog images\n";
    }

    return 0;
}


