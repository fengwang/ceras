#include "../include/ceras.hpp"

int main()
{
    using namespace ceras;
    random_generator.seed( 42 ); // fix random seed to reproduce the same result

    auto input = Input();
    auto output = sigmoid( Dense( 1, 2 )( input ) );
    auto m = model{ input, output };

    unsigned long const N = 1024;
    //auto cm = m.compile( CategoricalCrossentropy(), SGD(1, 1.0e-5f) );
    auto cm = m.compile( MeanAbsoluteError(), SGD(N, 1.0e-5f) );

    /*
    // define model, a single layer NN, using softmax activation
    auto x = place_holder<tensor<float>>{};
    auto W = variable{ tensor<float>{ {2, 2}, {1.0, -1.0, 1.0, -1.0} } };
    auto b = variable{ tensor<float>{{1,2}, {0.0, 0.0} } };
    auto p = softmax( x * W + b ); // p is our model
    */

    // preparing input for the model
    auto blues = randn<float>( {N, 2} ) - 2.0 * ones<float>( {N, 2} );
    auto reds = randn<float>( {N, 2} ) + 2.0 * ones<float>( {N, 2} );
    auto _x = concatenate( blues, reds, 0 );

    /*
    // binding input to layer x
    session<tensor<float>> s;
    s.bind( x, _x );
    */

    /*
    // define loss here
    auto c = place_holder<tensor<float>>{};
    auto J = cross_entropy( c, p );
    */

    // generating output/ground_truth for the model
    auto c_blue = tensor<float>{{1, 2}, {1.0, 0.0} };
    auto c_blues = repmat( c_blue, N, 1 );
    auto c_red = tensor<float>{{1, 2}, {0.0, 1.0} };
    auto c_reds = repmat( c_red, N, 1 );
    auto _c = concatenate( c_blues, c_reds, 0 );

    /*
    // binding output to the model
    s.bind( c, _c );
    // define optimizer here
    float const learning_rate = 1.0e-3;
    auto optimizer = gradient_descent{ J, 1, learning_rate }; // J is the loss, 1 is the batch size, learning_rate is the hyper-parameter
    */



    /*
    auto const iterations = 32UL;
    for ( auto idx = 0UL; idx != iterations; ++idx )
    {
        // first do forward propagation
        auto J_result = s.run( J );
        std::cout << "J at iteration " << idx+1 << ": " << J_result[0] << std::endl;
        // then do backward propagation
        s.run( optimizer );
    }
    */
    auto const iterations = 32UL;
    for ( auto idx = 0UL; idx != iterations; ++idx )
    {
        auto loss = cm.train_on_batch( _x, _c );
        std::cout << "Loss at step " << idx+1 << " : " << loss << std::endl;
    }

    return 0;
}

