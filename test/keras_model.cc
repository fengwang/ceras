#include "../include/keras/layer.hpp"
#include "../include/keras/model.hpp"

#include <iostream>

int main()
{
    using namespace Keras;
    auto input = Input( {28*28,} );
    auto layer_1 = Dense<512, activation<"relu">, use_bias<false>>{}( input );
    auto layer_2 = Dense<128, activation<"leaky_relu">>{}( layer_1 );
    auto layer_3 = Dense<32, activation<"relu">>{}( layer_2 );
    auto layer_4 = Dense<10>{}( layer_3 );

    auto model = Model{ input, layer_4 };
    //auto compiled_model = model.compile<loss<"crossentropy">, optimizer<"sgd", 32, "0.08">>();
    auto compiled_model = model.compile<optimizer<"sgd", 32, "0.08">, loss<"crossentropy">>();

    auto fake_inputs = ceras::random<float>( {32, 28*28} );
    auto fake_outputs = ceras::ones<float>( {32, 10} );

    //auto error = compiled_model.train_on_batch( fake_inputs, fake_outputs );
    auto errors = compiled_model.fit( fake_inputs, fake_outputs, 32, 10 );

    std::cout << "Got errors :\n";
    for ( auto error : errors ) std::cout << error << " ";
    std::cout << "\n";

    //model.fit<batch_size, epoch, split_ratio>( input_data, output_data );
    //model.train_on_batch( small_input, small_output );
    //model.predict( input_test );
    //model.save_weight( 'path_to_save.model' );
    //model.load_weight( 'path_to_save.model' );
    return 0;
}

