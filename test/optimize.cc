#include "../include/ceras.hpp"

int main()
{
    using namespace ceras;
    random_generator.seed( 42 ); // fix random seed to reproduce the same result

    // define model, a single layer NN, using softmax activation
    auto x = place_holder<tensor<double>>{};
    auto W = variable{ tensor<double>{ {2, 2}, {1.0, -1.0, 1.0, -1.0} } };
    auto b = variable{ tensor<double>{{1,2}, {0.0, 0.0} } };
    auto p = softmax( x * W + b ); // p is our model

    // preparing input for the model
    unsigned long const N = 512;
    auto blues = randn<double>( {N, 2} ) - 2.0 * ones<double>( {N, 2} );
    auto reds = randn<double>( {N, 2} ) + 2.0 * ones<double>( {N, 2} );
    auto _x = concatenate( blues, reds, 0 );

    // binding input to layer x
    session<tensor<double>> s;
    s.bind( x, _x );

    // define loss here
    auto c = place_holder<tensor<double>>{};
    auto J = cross_entropy( c, p );

    // generating output/ground_truth for the model
    auto c_blue = tensor<double>{{1, 2}, {1.0, 0.0} };
    auto c_blues = repmat( c_blue, N, 1 );
    auto c_red = tensor<double>{{1, 2}, {0.0, 1.0} };
    auto c_reds = repmat( c_red, N, 1 );
    auto _c = concatenate( c_blues, c_reds, 0 );

    // binding output to the model
    s.bind( c, _c );
    // define optimizer here
    double const learning_rate = 1.0e-3;
    auto optimizer = gradient_descent{ J, 1, learning_rate }; // J is the loss, 1 is the batch size, learning_rate is the hyper-parameter

    auto const iterations = 32UL;
    for ( auto idx = 0UL; idx != iterations; ++idx )
    {
        // first do forward propagation
        auto J_result = s.run( J );
        std::cout << "J at iteration " << idx+1 << ": " << J_result[0] << std::endl;
        // then do backward propagation
        s.run( optimizer );
    }

    return 0;
}

