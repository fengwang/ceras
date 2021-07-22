#include "../include/keras/layer.hpp"

int main()
{
    using namespace Keras;
    auto input = Input( {28*28,} );
    auto layer_1 = Dense<512, activation<"relu">, use_bias<false>>{}( input );
    auto layer_2 = Dense<128, activation<"leaky_relu">>{}( layer_1 );
    auto layer_3 = Dense<32, activation<"relu">>{}( layer_2 );
    auto layer_4 = Dense<10>{}( layer_3 );

    //auto model = Model( input, layer_4 );
    //auto compiled_model = model.compile<Loss<"cross_entropy">, Optimizer<"sgd">>();

    //model = Model( input, layer_4 );
    //model.compile< "cross_entropy", "sgd" >();
    //model.fit<batch_size, epoch, split_ratio>( input_data, output_data );
    //model.train_on_batch( small_input, small_output );
    //model.predict( input_test );
    //model.save_weight( 'path_to_save.model' );
    //model.load_weight( 'path_to_save.model' );
    return 0;
}

