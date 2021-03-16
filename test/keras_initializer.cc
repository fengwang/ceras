#include "../include/keras/initializer.hpp"

int main()
{
    using namespace Keras;

    {
        initializer<"random_normal"> init;
        auto v = init( {5, 5} );
        //std::cout << *(v.data_) << std::endl;
        std::cout << init.name_ << ": " <<  *(v.data_) << std::endl;
    }
    {
        initializer<"random_uniform"> init;
        auto v = init( {5, 5} );
        std::cout << init.name_ << ": " <<  *(v.data_) << std::endl;
    }
    {
        initializer<"truncated_normal", "0.0", "1.0", "-1.0", "1.0"> init;
        auto v = init( {5, 5} );
        std::cout << init.name_ << ": " <<  *(v.data_) << std::endl;
    }
    {
        initializer<"glorot_uniform"> init;
        auto v = init( {5, 5}, 64, 32 );
        std::cout << init.name_ << ": " <<  *(v.data_) << std::endl;
    }
    {
        initializer<"glorot_normal"> init;
        auto v = init( {5, 5}, 64, 32 );
        std::cout << init.name_ << ": " <<  *(v.data_) << std::endl;
    }
    {
        initializer<"constant", "1.23"> init;
        auto v = init( {5, 5} );
        std::cout << init.name_ << ": " <<  *(v.data_) << std::endl;
    }

    return 0;
}

