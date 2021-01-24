#include "../include/keras/layer.hpp"

void print( Keras::Input const& input )
{
    for ( auto const& v : input.shape_ )
    {
        if ( v )
            std::cout << v.value();
        else
            std::cout << "None";
        std::cout << " ";
    }
    std::cout << std::endl;
}

int main()
{
    using namespace Keras;
    {
        Input input_layer_0{ 1 };
        print( input_layer_0 );
        Input input_layer_1{ std::nullopt };
        print( input_layer_1 );
        Input input_layer_2{ 1, 2 };
        print( input_layer_2 );
        Input input_layer_3{ std::nullopt, 2 };
        print( input_layer_3 );
        Input input_layer_4{ std::nullopt, std::nullopt };
        print( input_layer_4 );
        Input input_layer_5{ std::nullopt, std::nullopt, 3 };
        print( input_layer_5 );
        Input input_layer_6{ std::nullopt, 3,  std::nullopt };
        print( input_layer_6 );
        Input input_layer_7{ 128, 128, 3 };
        print( input_layer_7 );
    }
    return 0;
}
