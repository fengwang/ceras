#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = Input(); // (4,)
    auto b = Dense( 11, 4 )( a );
    auto m1 = model{ a, b }; // 4->11

    auto x = Input(); // ( 7, )
    auto y = Dense( 4, 7 )( x );
    auto m2 = model{ x, y }; // 7->4

    auto input = Input();// // (7, )
    auto output = m1( m2( input ) ); // 7->4->11
    auto mm = model{ input, output };

    unsigned long const batch_size = 16;
    auto data = random<float>( {batch_size, 7} );

    auto output_data_1 = mm.predict( data );
    auto output_data_2 = m1.predict( m2.predict( data ) );

    auto diff = output_data_2 - output_data_1;

    std::cout <<  "The output of the composed model are:\n" << output_data_1 << std::endl;
    std::cout <<  "The difference between composed model and seperated models are:\n" << diff << std::endl;

    std::cout << "\nTesting model trainable method.\n";

    mm.trainable( false );
    auto output_data_3 = mm.predict( data );
    std::cout <<  "The output of the composed model at input 3, after setting non-trainable, are:\n" << output_data_3 << std::endl;

    mm.expression_.backward( random<float>( {batch_size, 7} ) );
    auto output_data_4 = mm.predict( data );
    std::cout <<  "The output of the composed model at input 4, after backpropagate random gradient, (should be the same as the previous output) are:\n" << output_data_4 << std::endl;
#if 0
    mm.trainable( true );
    mm.expression_.backward( random<float>( {batch_size, 7} ) );
    std::cout << "backward is done.\n";
    auto output_data_5 = mm.predict( data );
    std::cout << "prediction is done.\n";
    std::cout <<  "The output of the composed model at input 5, after setting trainable and backpropagate random gradient,  are:\n" << output_data_5 << std::endl;
#endif

    return 0;
}

