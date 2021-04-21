#include "../include/ceras.hpp"
#include <iostream>

int main()
{
    using namespace ceras;

    auto a = Input(); // (4,)
    auto b = Dense( 11, 4 )( a );
    auto m1 = model{ a, b }; // 4->16

    auto x = Input(); // ( 7, )
    auto y = Dense( 4, 7 )( x );
    auto m2 = model{ x, y }; // 7->4

    auto input = Input();// // (7, )
    auto output = m1( m2( input ) ); // 7->4->16
    auto mm = model{ input, output };

    unsigned long const batch_size = 16;
    auto data = random<float>( {batch_size, 7} );

    auto output_data_1 = mm.predict( data );
    auto output_data_2 = m1.predict( m2.predict( data ) );

    auto diff = output_data_2 - output_data_1;

    std::cout <<  "The output of the composed model are:\n" << output_data_1 << std::endl;
    std::cout <<  "The difference between composed model and seperated models are:\n" << diff << std::endl;



    return 0;
}

