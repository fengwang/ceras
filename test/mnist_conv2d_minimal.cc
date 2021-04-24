#include "../include/ceras.hpp"
#include <iostream>
int main()
{
    using namespace ceras;
    random_generator.seed( 42 );

    auto input = Input(); // shape( 28, 28 )
    auto l0 = Reshape({28, 28, 1})( input );
    auto l1 = ReLU( Conv2D( 32, {3, 3}, {28, 28, 1}, "same" )(l0) );
    auto l2 = MaxPooling2D( 2 )( l1 );
    auto l3 = ReLU( Conv2D( 64, {3, 3}, {14, 14, 32}, "same" )(l2) );
    auto l4 = MaxPooling2D( 2 )( l3 ); // 7, 7, 64
    auto l5 = Flatten()( l4 );
    auto l6 = ReLU( Dense( 128, 7*7*64 )( l5 ) );
    auto output = Dense( 10, 128 )( l6 );
    auto m = model( input, output );

    std::size_t const batch_size = 10;
    float learning_rate = 0.005f;
    auto cm = m.compile( CategoricalCrossentropy(), SGD(batch_size, learning_rate) );

    unsigned long epoches = 10;
    int verbose = 1;
    double validation_split = 0.1;
    auto const& [x_training, y_training, x_test, y_test] = dataset::mnist::load_data();

    auto history = cm.fit( x_training.as_type<float>()/255.0f, y_training.as_type<float>(), batch_size, epoches, verbose, validation_split );

    auto error = cm.evaluate( x_test.as_type<float>()/255.0, y_test.as_type<float>(), batch_size );

    std::cout << "\nPrediction error on the test set is " << error << std::endl;

    return 0;
}

