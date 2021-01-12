#include "../include/ceras.hpp"

int main()
{
    using namespace ceras;

    {
        std::cerr << "Test Case of Opimizer " << 1 << std::endl;
        auto x = place_holder<tensor<double>>{};
        auto c = place_holder<tensor<double>>{};
        auto W = variable{ tensor<double>{ {2, 2}, {1.0, -1.0, 1.0, -1.0} } };
        auto b = variable{ tensor<double>{{1,2}, {0.0, 0.0} } };

        auto p = softmax( x * W + b );

        unsigned long const N = 512;
        auto blues = randn<double>( {N, 2} ) - 2.0 * ones<double>( {N, 2} );
        auto reds = randn<double>( {N, 2} ) + 2.0 * ones<double>( {N, 2} );
        auto _x = concatenate( blues, reds, 0 );

        //std::cout << "X is\n" << _x << std::endl;

        session<tensor<double>> s;
        s.bind( x, _x );
        auto result = s.run( p );
        std::cout << "Result of p:\n" << result << std::endl;

        //auto J = negative( sum_reduce(elementwise_multiply( c, ceras::log(p) )) );
        auto J = cross_entropy( c, p );

        auto c_blue = tensor<double>{{1, 2}, {1.0, 0.0} };
        auto c_blues = repmat( c_blue, N, 1 );
        auto c_red = tensor<double>{{1, 2}, {0.0, 1.0} };
        auto c_reds = repmat( c_red, N, 1 );
        auto _c = concatenate( c_blues, c_reds, 0 );

        s.bind( c, _c );
        //auto J_result = s.run( J );
        //std::cout << "Result of J:\n" << J_result << std::endl;
        double const learning_rate = 1.0e-3;
        auto optimizer = gradient_descent{ J, 1, learning_rate };

        auto const iterations = 32UL;
        for ( auto idx = 0UL; idx != iterations; ++idx )
        {
            //debug_print("Iteration started at ", idx);
            auto J_result = s.run( J );
            std::cout << "J at iteration " << idx+1 << ": " << J_result[0] << std::endl;
            s.run( optimizer );
            //debug_print("Iteration finished at ", idx);
        }

    }

    return 0;
}

