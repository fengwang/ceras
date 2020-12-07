#include "../include/ceras.hpp"
#include "../include/utils/range.hpp"

int main()
{
    using namespace ceras;

    {
        std::cerr << "Test Case of Xor " << 1 << std::endl;

        auto input = place_holder<double>{}; // Nx2

        auto w1 = variable<double>{ ceras::numeric::randn<double>( {2, 13} ) };
        auto b1 = variable<double>{ ceras::numeric::randn<double>( {1, 13} ) };
        //auto l1 = sigmoid( input * w1 + b1 );
        auto l1 = ceras::tanh( input * w1 + b1 );

        auto w2 = variable<double>{ ceras::numeric::randn<double>( {13, 1} ) };
        auto b2 = variable<double>{ ceras::numeric::randn<double>( {1, 1} ) };

        auto p = sigmoid( l1 * w2 + b2 ); //output

        auto output = place_holder<double>{};

        // mse error
        auto loss = square_loss( p, output );

        // XOR dataset
        ceras::numeric::tensor<double> inputs{ {4, 2}, {0.0, 0.0,
                                                        0.0, 1.0,
                                                        1.0, 0.0,
                                                        1.0, 1.0 } };
        ceras::numeric::tensor<double> outputs{ {4, 1}, {0.0, 1.0, 1.0, 0.0 }};

        // prepare the training
        session<double> s;
        s.bind( input, inputs );
        s.bind( output, outputs );

        double const learning_rate = 1.0e-2;
        auto optimizer = gradient_descent{ loss, learning_rate };

        //auto const iterations = 1024UL;
        auto const iterations = 32*1024UL;
        for ( auto idx : ceras::range( iterations ) )
        {
            auto current_error = s.run( loss );
            std::cout << "error at iteration " << idx << ":\t" << current_error[0] << std::endl;
            s.run( optimizer );
        }

    }

    return 0;
}

