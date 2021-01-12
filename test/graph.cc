#include "../include/ceras.hpp"

int main()
{
    using namespace ceras;
    //using namespace ceras::numeric;

    tensor<double> _A{{2, 2}, {1.0, 0.0, 0.0, -1.0}};
    tensor<double> _b{{2, 1}, {1.0, 1.0}};

    variable A{_A};
    variable b{_b};
    place_holder<tensor<double>> x;

    auto z = A * x + b;

    {
        std::cerr << "Test Case " << 1 << std::endl;
        session<tensor<double>> s;
        tensor<double> _x{ {2, 1}, {1, 2} };
        s.bind( x, _x );
        auto result = s.run( z );
        std::cout << "Result:\n" << result << std::endl;
    }
    if (0) //failure case
    {
        std::cerr << "Test Case " << 2 << std::endl;
        session<tensor<double>> s;
        auto result = s.run( z );
        std::cout << "Result:\n" << result << std::endl;
    }

    auto sz = sigmoid( z );
    {
        std::cerr << "Test Case " << 3 << std::endl;
        session<tensor<double>> s;
        tensor<double> _x{ {2, 1}, {1, 2} };
        s.bind( x, _x );
        auto result = s.run( sz );
        std::cout << "Result:\n" << result << std::endl;
    }

    {
        std::cerr << "Test Case " << 4 << std::endl;
        auto x = place_holder<tensor<double>>{};
        auto w = variable{ tensor<double>{{1,3}, {1.0, 1.0, 1.0}} };
        auto b = variable{ tensor<double>{{1, 1}, {0.0}} };
        auto p = sigmoid( w * x + b );
        auto _x = tensor<double>{ {3, 1}, {3.0, 2.0, 1.0} };
        session<tensor<double>> s;
        s.bind( x, _x );
        auto result = s.run( p );
        std::cout << "Result:\n" << result << std::endl;
    }
    {
        std::cerr << "Test Case " << 5 << std::endl;
        auto x = place_holder<tensor<double>>{};
        auto c = place_holder<tensor<double>>{};
        auto W = variable{ tensor<double>{ {2, 2}, {1.0, -1.0, 1.0, -1.0} } };
        auto b = variable{ tensor<double>{{1,2}, {0.0, 0.0} } };

        auto p = softmax( x * W + b );

        unsigned long const N = 5;
        auto blues = randn<double>( {N, 2} ) - 2.0 * ones<double>( {N, 2} );
        auto reds = randn<double>( {N, 2} ) + 2.0 * ones<double>( {N, 2} );
        auto _x = concatenate( blues, reds, 0 );

        std::cout << "X is\n" << _x << std::endl;

        session<tensor<double>> s;
        s.bind( x, _x );
        auto result = s.run( p );
        std::cout << "Result of p:\n" << result << std::endl;

        auto J = negative( sum_reduce(elementwise_multiply( c, ceras::log(p) )) );

        auto c_blue = tensor<double>{{1, 2}, {1.0, 0.0} };
        auto c_blues = repmat( c_blue, N, 1 );
        auto c_red = tensor<double>{{1, 2}, {0.0, 1.0} };
        auto c_reds = repmat( c_red, N, 1 );
        auto _c = concatenate( c_blues, c_reds, 0 );

        s.bind( c, _c );
        auto J_result = s.run( J );
        std::cout << "Result of J:\n" << J_result << std::endl;
    }
    {
        std::cerr << "Test Case " << 6 << std::endl;
        auto x = place_holder<tensor<double>>{};
        auto c = place_holder<tensor<double>>{};
        auto W = variable{ tensor<double>{ {2, 2}, {1.0, -1.0, 1.0, -1.0} } };
        auto b = variable{ tensor<double>{{1,2}, {0.0, 0.0} } };

        auto p = softmax( x * W + b );

        auto Ws = variable{ tensor<double>{ {2, 2}, {1.0, -1.0, 1.0, -1.0} } };
        auto bs = variable{ tensor<double>{{1,2}, {-0.01, 0.01} } };
        auto ps = softmax( x * Ws + bs );

        auto pp = sigmoid( sigmoid( p + ps ) + p );

        unsigned long const N = 5;
        auto blues = randn<double>( {N, 2} ) - 2.0 * ones<double>( {N, 2} );
        auto reds = randn<double>( {N, 2} ) + 2.0 * ones<double>( {N, 2} );
        auto _x = concatenate( blues, reds, 0 );

        std::cout << "X is\n" << _x << std::endl;

        session<tensor<double>> s;
        s.bind( x, _x );
        auto result = s.run( pp );
        std::cout << "Result of pp:\n" << result << std::endl;

        auto J = negative( sum_reduce(elementwise_multiply( c, ceras::log(pp) )) );

        auto c_blue = tensor<double>{{1, 2}, {1.0, 0.0} };
        auto c_blues = repmat( c_blue, N, 1 );
        auto c_red = tensor<double>{{1, 2}, {0.0, 1.0} };
        auto c_reds = repmat( c_red, N, 1 );
        auto _c = concatenate( c_blues, c_reds, 0 );

        s.bind( c, _c );
        auto J_result = s.run( J );
        std::cout << "Result of J:\n" << J_result << std::endl;

    }
    return 0;
}

