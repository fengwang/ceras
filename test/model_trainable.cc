#include "../include/ceras.hpp"

int main()
{
    using namespace ceras;
    random_generator.seed( 42 ); // fix random seed to reproduce the same result

    auto input = Input();
    auto output = sigmoid( Dense( 1, 2 )( input ) );
    auto m = model{ input, output };

    unsigned long const N = 1024 * 32;
    auto cm = m.compile( MeanAbsoluteError(), SGD(N, 1.0e-1f) );

    // preparing input for the model
    auto blues = randn<float>( {N, 2} ) - 2.0 * ones<float>( {N, 2} );
    auto reds = randn<float>( {N, 2} ) + 2.0 * ones<float>( {N, 2} );
    auto _x = concatenate( blues, reds, 0 );

    // generating output/ground_truth for the model
    auto c_blue = tensor<float>{{1, 2}, {1.0, 0.0} };
    auto c_blues = repmat( c_blue, N, 1 );
    auto c_red = tensor<float>{{1, 2}, {0.0, 1.0} };
    auto c_reds = repmat( c_red, N, 1 );
    auto _c = concatenate( c_blues, c_reds, 0 );

    auto const iterations = 128UL;
    for ( auto idx = 0UL; idx != iterations; ++idx )
    {
        auto loss = cm.train_on_batch( _x, _c );
        //std::cout << "Loss at step " << idx+1 << " : " << loss << std::endl;
    }

    auto loss_0 = cm.evaluate( _x, _c );
    std::cout << "\nAfter training the loss is " << loss_0 << std::endl;

    cm.trainable( false );
    std::cout << "\nStop the model from trainable." << std::endl;

    //auto const iterations = 32UL;
    for ( auto idx = 0UL; idx != iterations; ++idx )
    {
        auto loss = cm.train_on_batch( _x, _c );
        //std::cout << "Loss at step " << idx+1 << " : " << loss << std::endl;
    }

    auto loss_1 = cm.evaluate( _x, _c );
    std::cout << "\nEvaluating the loss again:" << loss_1 << std::endl;


    cm.trainable( true );
    std::cout << "\nStart the model from trainable." << std::endl;

    //auto const iterations = 32UL;
    for ( auto idx = 0UL; idx != iterations; ++idx )
    {
        auto loss = cm.train_on_batch( _x, _c );
        //std::cout << "Loss at step " << idx+1 << " : " << loss << std::endl;
    }

    auto loss_2 = cm.evaluate( _x, _c );
    std::cout << "\nEvaluating the 3rd time:" << loss_2 << std::endl;

    return 0;
}

