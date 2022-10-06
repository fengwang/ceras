#include "../include/ceras.hpp"
#include <iostream>



//(A \* a) .*  (B \* b) * c -> C

int main()
{
    using namespace ceras;
    random_generator.seed( 113 );
    constexpr unsigned long m = 2;
    constexpr unsigned long ops = 7; // 7->6->5->4
    constexpr unsigned long epochs = 1024;
    constexpr unsigned training_samples = 1024;
    constexpr unsigned long iterations = 1024;
    float learning_rate = 0.025f;


    // prepare traing data
    tensor<float> train_A = rand<float>( {training_samples, m*m} );
    auto train_A_v = view<float, 3>{ train_A.data(), training_samples, m, m };
    tensor<float> train_B = rand<float>( {training_samples, m*m} );
    auto train_B_v = view<float, 3>{ train_B.data(), training_samples, m, m };
    tensor<float> train_C = zeros<float>( {training_samples, m*m} );
    auto train_C_v = view<float, 3>{ train_C.data(), training_samples, m, m };
    for ( auto idx : range( training_samples ) )
    {
        for ( auto row : range( m ) )
            for ( auto k : range( m ) )
                for ( auto col : range( m ) )
                    train_C_v[idx][row][col] += train_A_v[idx][row][k] * train_B_v[idx][k][col];
    }

    //session<tensor<float>> s;
    auto& s = get_default_session<tensor<float>>();

    auto A = place_holder<tensor<float>>();
    s.bind( A, train_A );
    auto a = variable{ randn<float>({m*m, ops}) };
    auto _a = variable{ randn<float>({m*m, ops}) };

    auto B = place_holder<tensor<float>>();
    s.bind( B, train_B );
    auto b = variable{ randn<float>({m*m, ops}) };
    auto _b = variable{ randn<float>({m*m, ops}) };

    auto C = place_holder<tensor<float>>();
    s.bind( C, train_C );
    auto c = variable{ randn<float>({ops, m*m}) };
    auto _c = variable{ randn<float>({ops, m*m}) };

    for ( auto it : range( iterations ) )
    {
        value<float> alpha{ 1.0f + static_cast<float>(20 * it / iterations) };

        auto AA = A * elementwise_product( tanh( alpha * a ), sigmoid( alpha * _a ) ); // shape ( 1, ops )
        auto BB = B * elementwise_product( tanh( alpha * b ), sigmoid( alpha * _b ) ); // shape ( 1, ops )
        auto AB = elementwise_product( AA, BB ); // shape (1, ops ), real multiplication happens here
        auto CC = AB * elementwise_product( tanh( alpha * c ), sigmoid( alpha * _c ) ); // shape ( 1, ops )
        auto loss = mse( CC, C );
        auto optimizer = sgd{ loss, training_samples, learning_rate };
        for ( auto e : range( epochs ) )
        {
            auto current_error = s.run( loss );
            s.run( optimizer );
            std::cout << "Loss at iteration " << it << ", epoch " << e << ": " << current_error << std::endl;
        }
    }

    return 0;
}

