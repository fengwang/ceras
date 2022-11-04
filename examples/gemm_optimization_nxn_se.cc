// ./bin/test_gemm_optimization_nxn -m 2 -ops 7 -epochs 1000 -training_samples 1023 -iteration 1025 -learning_rate 1.0
#include "../include/ceras.hpp"
#include "./parser.hpp"
#include "../include/utils/fmt.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>

inline int send_message( std::string const& message )
{
    return 0;
}


//(A \* a) .*  (B \* b) * c -> C

int main( int argc, char** argv )
{
    using namespace ceras;
    random_generator.seed( 113 );
    constexpr float stop_threshold = 1.0e-7f;
    constexpr float success_threshold = 1.0e-3f;
    constexpr unsigned long final_iterations = 128;

    unsigned long scale = 16; // search in [-scale, scale]
    //unsigned long scale = 2; // search in [-scale, scale]
    unsigned long m = 2;
    unsigned long ops = 7;
    unsigned long epochs = 1024*4;
    unsigned long training_samples = 1024*1024;
    unsigned long iterations = 1024;
    //float learning_rate = 1.0f;
    float learning_rate = 1.0e-2f;

    auto p_r = parser::make_option<unsigned long>( "-s", [&scale]( unsigned long _m ){ scale = _m; }  );
    auto p_m = parser::make_option<unsigned long>( "-m", [&m]( unsigned long _m ){ m = _m; }  );
    auto p_ops = parser::make_option<unsigned long>( "-ops", [&ops]( unsigned long _ops ){ ops = _ops; }  );
    auto p_epochs = parser::make_option<unsigned long>( "-epochs", [&epochs]( unsigned long _epochs ){ epochs = _epochs; }  );
    auto p_training_samples = parser::make_option<unsigned long>( "-training_samples", [&training_samples]( unsigned long _training_samples ){ training_samples = _training_samples; }  );
    auto p_iterations = parser::make_option<unsigned long>( "-iterations", [&iterations]( unsigned long _iterations ){ iterations = _iterations; }  );
    auto p_learning_rate = parser::make_option<float>( "-learning_rate", [&learning_rate]( float _learning_rate ){ learning_rate = _learning_rate; }  );

    parser::parse( argc, argv, p_m, p_ops, p_epochs, p_training_samples, p_iterations, p_learning_rate );
    {
        std::cout << "Configuration: \n\n";
        std::cout << "m = \t" << m << "\n";
        std::cout << "ops = \t" << ops << "\n\n";
        std::cout << "epochs = \t" << epochs << "\n";
        std::cout << "training_samples = \t" << training_samples << "\n";
        std::cout << "iterations = \t" << iterations << "\n";
        std::cout << "learning_rate = \t" << learning_rate << "\n";


        std::string const report = fmt::format( "Running gemm optimization with m={}, ops={}, epochs={}, training_samples={}, iterations={} and learning_rate={}", m, ops, epochs, training_samples, iterations, learning_rate );
        std::cout << "Generated report:\n" << report << "\n";
        send_message( report );

    }

    // prepare traing data
    tensor<float> train_A = rand<float>( {training_samples, m*m} );
    auto train_A_v = view<float, 3>{ train_A.data(), training_samples, m, m };
    tensor<float> train_B = rand<float>( {training_samples, m*m} );
    auto train_B_v = view<float, 3>{ train_B.data(), training_samples, m, m };
    tensor<float> train_C = zeros<float>( {training_samples, m*m} );
    auto train_C_v = view<float, 3>{ train_C.data(), training_samples, m, m };
    for ( auto idx : ceras::range( training_samples ) )
    {
        for ( auto row : ceras::range( m ) )
            for ( auto k : ceras::range( m ) )
                for ( auto col : ceras::range( m ) )
                    train_C_v[idx][row][col] += train_A_v[idx][row][k] * train_B_v[idx][k][col];
    }

    auto& s = get_default_session<tensor<float>>();

    auto ones_mmop = variable{ ones<float>({m*m, ops}) };
    auto ones_opmm = variable{ ones<float>({ops, m*m}) };

    auto A = place_holder<tensor<float>>();
    s.bind( A, train_A );

    auto tanh_a_ = variable{ randn<float>({m*m, scale}, 0.0f, 1.0f/std::sqrt(m*m*scale*1.0f)) };
    auto sigmoid_a_ = variable{ randn<float>({m*m, scale}, 0.0f, 1.0f/std::sqrt(m*m*scale*1.0f)) };

    auto tanh__a = variable{ randn<float>({scale, ops}, 0.0f, 1.0f/std::sqrt(scale*ops*1.0f)) };
    auto sigmoid__a = variable{ randn<float>({scale, ops}, 0.0f, 1.0f/std::sqrt(scale*ops*1.0f)) };

    auto B = place_holder<tensor<float>>();
    s.bind( B, train_B );

    auto tanh_b_ = variable{ randn<float>({m*m, scale}, 0.0f, 1.0f/std::sqrt(m*m*scale*1.0f)) };
    auto sigmoid_b_ = variable{ randn<float>({m*m, scale}, 0.0f, 1.0f/std::sqrt(m*m*scale*1.0f)) };

    auto tanh__b = variable{ randn<float>({scale, ops}, 0.0f, 1.0f/std::sqrt(scale*ops*1.0f)) };
    auto sigmoid__b = variable{ randn<float>({scale, ops}, 0.0f, 1.0f/std::sqrt(scale*ops*1.0f)) };

    auto C = place_holder<tensor<float>>();
    s.bind( C, train_C );

    auto tanh__c = variable{ randn<float>({ops, scale}, 0.0f, 1.0f/std::sqrt(ops*scale*1.0f)) };
    auto sigmoid__c = variable{ randn<float>({ops, scale}, 0.0f, 1.0f/std::sqrt(ops*scale*1.0f)) };

    auto tanh_c_ = variable{ randn<float>({scale, m*m}, 0.0f, 1.0f/std::sqrt(scale*m*m*1.0f)) };
    auto sigmoid_c_ = variable{ randn<float>({scale, m*m}, 0.0f, 1.0f/std::sqrt(scale*m*m*1.0f)) };

    bool found_flag = false;

    for ( auto it : ceras::range( iterations ) )
    {
        if (found_flag)
            break;

        //value<float> alpha{ 1.0f + static_cast<float>(1000.0 * it*it / (iterations*iterations)) };
        value<float> alpha{ 1.0f + 0.1f*it };

        // [N, ops] = [N, m^2] * [[m^2, scale] * [scale, ops]]
        auto AA = A * ( elementwise_product( tanh( alpha * tanh_a_ ), sigmoid( alpha * sigmoid_a_ ) ) *
                        elementwise_product( tanh( alpha * tanh__a ), sigmoid( alpha * sigmoid__a ) ) );
        // [N, ops] = [N, m^2] * [[m^2, scale] * [scale, ops]]
        auto BB = B * ( elementwise_product( tanh( alpha * tanh_b_ ), sigmoid( alpha * sigmoid_b_ ) ) *
                        elementwise_product( tanh( alpha * tanh__b ), sigmoid( alpha * sigmoid__b ) ) );
        // [N, ops]
        auto AB = elementwise_product( AA, BB );

        // [N, m^2] =  [N, ops] * [[op, scale] * [scale, m^2]]
        auto CC = AB * ( elementwise_product( tanh( alpha * tanh__c ), sigmoid( alpha * sigmoid__c ) ) *
                         elementwise_product( tanh( alpha * tanh_c_ ), sigmoid( alpha * sigmoid_c_ ) ) );
        auto loss = mse( CC, C );

        //auto optimizer = adam{ loss, training_samples, learning_rate*static_cast<float>(1.0-it/iterations) };
        auto optimizer = adam{ loss, 1, learning_rate*static_cast<float>(1.0-it/iterations) };
        for ( auto e : ceras::range( epochs ) )
        {
            auto current_error = s.run( loss );
            s.run( optimizer );
            if (current_error[0] <= stop_threshold && (it >= 490) ) // sigmoid of 50
            {
                found_flag = true;
                std::cout << "Maybe found with loss = " << current_error[0] << std::endl;
                std::cout << "BREAK!" << std::endl;
                break;
            }
            std::cout.precision( 8 );
            std::cout << "\33[2K\r";
            std::cout << "\rLoss at iteration\t" << it << ", epoch\t" << e << ":\t" << current_error[0] << std::flush;
            if (current_error[0]<stop_threshold)
            {
                std::cout << "\nError is already small enough, early stop at current iteration." << std::endl;
                break;
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;


    {
        value<float> alpha{ 1.0e2f };

        //auto AA = A * elementwise_product( tanh( alpha * a ), sigmoid( alpha * _a ) ); // shape ( 1, ops )
        auto AA = A * ( elementwise_product( tanh( alpha * tanh_a_ ), sigmoid( alpha * sigmoid_a_ ) ) *
                        elementwise_product( tanh( alpha * tanh__a ), sigmoid( alpha * sigmoid__a ) ) );
        //auto BB = B * elementwise_product( tanh( alpha * b ), sigmoid( alpha * _b ) ); // shape ( 1, ops )
        auto BB = B * ( elementwise_product( tanh( alpha * tanh_b_ ), sigmoid( alpha * sigmoid_b_ ) ) *
                        elementwise_product( tanh( alpha * tanh__b ), sigmoid( alpha * sigmoid__b ) ) );
        auto AB = elementwise_product( AA, BB ); // shape (1, ops ), real multiplication happens here
        //auto CC = AB * elementwise_product( tanh( alpha * c ), sigmoid( alpha * _c ) ); // shape ( 1, ops )
        auto CC = AB * ( elementwise_product( tanh( alpha * tanh__c ), sigmoid( alpha * sigmoid__c ) ) *
                         elementwise_product( tanh( alpha * tanh_c_ ), sigmoid( alpha * sigmoid_c_ ) ) );
        auto loss = mse( CC, C );
        auto optimizer = adam{ loss, training_samples, 1.0e-10f };

        for ( [[maybe_unused]] auto idx : ceras::range( final_iterations ) )
        {
            s.run( loss );
            s.run( optimizer );
        }

        auto current_error = s.run( loss );
        std::cout << "The final error is : " << current_error[0] << std::endl;
        send_message( fmt::format("Finished with error: {}.\n", current_error[0]) );

        //if ( (current_error[0] > success_threshold) && (!found_flag) ) return 0;

        if (current_error[0] >= success_threshold)
        {
            std::cout << "Failed to solve with a big error of " << current_error[0] << std::endl;
            //return 0;
        }

        {
            auto op = ( elementwise_product( tanh( alpha * tanh_a_ ), sigmoid( alpha * sigmoid_a_ ) ) *
                        elementwise_product( tanh( alpha * tanh__a ), sigmoid( alpha * sigmoid__a ) ) );
            auto _ = s.run( op );
            std::cout << "AA is \n" << _ << std::endl;

            send_message( fmt::format( "AA = {}\n", _ ) );
        }

        {
            auto op = ( elementwise_product( tanh( alpha * tanh_b_ ), sigmoid( alpha * sigmoid_b_ ) ) *
                        elementwise_product( tanh( alpha * tanh__b ), sigmoid( alpha * sigmoid__b ) ) );
            auto _ = s.run( op );
            std::cout << "BB is \n" << _ << std::endl;

            send_message( fmt::format( "BB = {}\n", _ ) );
        }

        {
            auto op = ( elementwise_product( tanh( alpha * tanh__c ), sigmoid( alpha * sigmoid__c ) ) *
                         elementwise_product( tanh( alpha * tanh_c_ ), sigmoid( alpha * sigmoid_c_ ) ) );
            auto _ = s.run( op );
            std::cout << "CC is \n" << _ << std::endl;

            send_message( fmt::format( "CC = {}\n", _ ) );
        }

    }

#if 0
The final error is : 0.082381748
Failed to solve with a big error of 0.082381748
AA is
shape: [ 4 7 ]
data:
{
-1      1.2904848e-12   0.23105203      -0.20029597     -1.0184351      -0.00016394258  -0.030535474
0.99391317      -2.2744966e-10  -1.249577       -0.53431404     -0.8774662      -0.0079396963   -0.015919933
-0.99546152     -2.0218768e-12  1.1920929e-07   -0.28524923     1.001645        0.99983239      -0.12638406
1.0799493       1.2473089       1.1156967       1.1156965       0.87390894      -1.0889298      -0.012156788
}

BB is
shape: [ 4 7 ]
data:
{
-0.952806       1.0879598e-09   1.0002449       -0.96508837     0.44325101      1.8792939e-10   -1.9847987
0.81507289      -0.99884892     0.021016836     -0.93730348     0.11586368      -1      0.98638427
1.0471959       0.00039440684   -1.5374367      1.5742464       0.0018293262    1.0135657e-13   -1.9999456
-0.86225557     -1.0015978      -0.022960745    -0.67190945     -0.0020945072   -8.5339912e-12  0.9984144
}

CC is
shape: [ 7 4 ]
data:
{
0.018132448     -0.98338282     -1.1635064e-16  7.3590786e-12
-0.18657196     -0.81341422     -0.14051415     -0.86660081
0.44397047      0.9993223       -1.1920929e-07  1.4603708e-07
0.16061521      1.9847407       0.70693874      1.9800396e-07
-0.94688904     -2.3841858e-07  1.0000006       8.2610316e-12
-0.0196041      0.9991408       0.0099228024    -1.0002015
1.0001765       0       1.0081941       5.9604645e-08
}

#endif

    return 0;
}

