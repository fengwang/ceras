#include "../include/ceras.hpp"
#include <iostream>
#include <cmath>



//(A \* a) .*  (B \* b) * c -> C

int main()
{
    using namespace ceras;
    random_generator.seed( 113 );
    //constexpr unsigned long m = 2;
    //constexpr unsigned long ops = 7; // 7->6->5->4
    constexpr unsigned long m = 4;
    constexpr unsigned long ops = 46;
    constexpr unsigned long epochs = 1024*4;
    constexpr unsigned training_samples = 1024;
    constexpr unsigned long iterations = 1024;
    constexpr float stop_threshold = 1.0e-8f;
    float learning_rate = 1.0f;


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

    auto ones_mmop = variable{ ones<float>({m*m, ops}) };
    auto ones_opmm = variable{ ones<float>({ops, m*m}) };

    auto A = place_holder<tensor<float>>();
    s.bind( A, train_A );
    auto a = variable{ randn<float>({m*m, ops}, 0.0f, 1.0f) };
    auto _a = variable{ randn<float>({m*m, ops}, 0.0f, 1.0f) };

    auto B = place_holder<tensor<float>>();
    s.bind( B, train_B );
    auto b = variable{ randn<float>({m*m, ops}, 0.0f, 1.0f) };
    auto _b = variable{ randn<float>({m*m, ops}, 0.0f, 1.0f) };

    auto C = place_holder<tensor<float>>();
    s.bind( C, train_C );
    auto c = variable{ randn<float>({ops, m*m}, 0.0f, 1.0f) };
    auto _c = variable{ randn<float>({ops, m*m}, 0.0f, 1.0) };

    bool maybe_found = false;

    for ( auto it : range( iterations ) )
    {
        if (maybe_found)
            break;

        value<float> alpha{ 1.0f + static_cast<float>(1000.0 * it*it / (iterations*iterations)) };
        value<float> beta{ 1.0e-4f * (1.0f - static_cast<float>(it/iterations)) };

        auto AA = A * elementwise_product( tanh( alpha * a ), sigmoid( alpha * _a ) ); // shape ( 1, ops )
        auto BB = B * elementwise_product( tanh( alpha * b ), sigmoid( alpha * _b ) ); // shape ( 1, ops )
        auto AB = elementwise_product( AA, BB ); // shape (1, ops ), real multiplication happens here
        auto CC = AB * elementwise_product( tanh( alpha * c ), sigmoid( alpha * _c ) ); // shape ( 1, ops )
        auto loss = mse( CC, C );
        //auto loss = mae( CC, C );

        auto optimizer = adam{ loss, training_samples, learning_rate*static_cast<float>(1.0-it/iterations) };
        for ( auto e : range( epochs ) )
        {
            auto current_error = s.run( loss );
            s.run( optimizer );
            if (current_error[0] <= stop_threshold)
            {
                maybe_found = true;
                std::cout << "Maybe found with loss = " << current_error[0] << std::endl;
                std::cout << "BREAK!" << std::endl;
                break;
            }
            std::cout << "Loss at iteration " << it << ", epoch " << e << ": " << current_error[0] << std::endl;
        }
    }


    {
        value<float> alpha{ 1.0e3f };

        auto AA = A * elementwise_product( tanh( alpha * a ), sigmoid( alpha * _a ) ); // shape ( 1, ops )
        auto BB = B * elementwise_product( tanh( alpha * b ), sigmoid( alpha * _b ) ); // shape ( 1, ops )
        auto AB = elementwise_product( AA, BB ); // shape (1, ops ), real multiplication happens here
        auto CC = AB * elementwise_product( tanh( alpha * c ), sigmoid( alpha * _c ) ); // shape ( 1, ops )
        auto loss = mse( CC, C );
        //auto loss = mae( CC, C );
        auto optimizer = adam{ loss, training_samples, 1.0e-5f };

        for ( auto idx : range( 16 ) )
        {
            s.run( loss );
            s.run( optimizer );
        }
        auto current_error = s.run( loss );
        std::cout << "The final error is : " << current_error << std::endl;

        {
            auto op = elementwise_product( tanh( alpha * a ), sigmoid( alpha * _a ) );
            auto _ = s.run( op );
            std::cout << "AA is \n" << _ << std::endl;
        }

        {
            auto op = elementwise_product( tanh( alpha * b ), sigmoid( alpha * _b ) );
            auto _ = s.run( op );
            std::cout << "BB is \n" << _ << std::endl;
        }

        {
            auto op = elementwise_product( tanh( alpha * c ), sigmoid( alpha * _c ) );
            auto _ = s.run( op );
            std::cout << "CC is \n" << _ << std::endl;
        }

    }


    return 0;
}

