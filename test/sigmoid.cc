#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include <cmath>

TEST_CASE("sigmoid", "[sigmoid]")
{
    using namespace ceras;

    unsigned long const tests = 1024;
    unsigned long const upper_dims = 7;
    //unsigned long const tests = 1;
    //unsigned long const upper_dims = 2;

    std::vector<unsigned long> vdims;
    for ( auto idx : range( upper_dims ) )
        vdims.push_back( idx+1 );

    for ( auto _ : range( tests ) )
    {
        for ( unsigned long dims_ = 0UL; dims_ != upper_dims; ++dims_ )
        {
            unsigned long dims = dims_ + 2;

            auto _x = numeric::random<double>( {1, dims} );
            auto tw = numeric::random<double>( {dims, dims} );
            auto tb = numeric::random<double>( {1, 1} );
            //auto _x = numeric::ones<double>( {1, dims} );
            //auto tw = numeric::ones<double>( {dims, dims} );
            //auto tb = numeric::ones<double>( {1, 1} );

            auto x = place_holder<double>{};
            auto c = place_holder<double>{};
            auto W = variable<double>{ tw };
            auto b = variable<double>{ tb };

            auto p = sigmoid( x * W + b );

            session<double> s;
            s.bind( x, _x );
            auto result = s.run( p );

            auto result_ = _x * tw + tb;
            result_.map( []( double x ){  return 1.0 / ( 1.0 + std::exp(-x) ); } );

            auto df = result - result_;

            for ( auto idx : range( df.size() ) )
                REQUIRE( std::abs( df[idx] ) < 1.0e-7 );


            auto ground_truth = place_holder<double>{};
            //auto loss = cross_entropy( ground_truth, p );
            //auto loss = square_loss( ground_truth, p );
            auto loss = sum_reduce( square( minus( ground_truth, p ) ) );


            auto target = numeric::random<double>( {1, dims} );
            //auto target = numeric::ones<double>( {1, dims} );

            s.bind( ground_truth, target );

            double learning_rate = 1.0e-14;
            double momentum = 0.0;
            auto optimizer = gradient_descent{ loss, learning_rate, momentum };

            auto w_data = (*(W.data_)).deep_copy();
            auto b_data = (*(b.data_)).deep_copy();

            auto residual = s.run( loss );
            s.run( optimizer );

            // todo: check gradients for variable W and b
            auto w_gradient = (*(W.gradient_)).deep_copy();
            //std::cout << "Generated weight gradient:\n" << w_gradient << std::endl;
            auto b_gradient = (*(b.gradient_)).deep_copy();

            // check gradients
            auto i0 = x;
            auto i1 = W;
            auto i2 = x * W;
            auto i3 = b;
            auto i4 = i2 + i3;
            auto i5 = sigmoid( i4 );
            auto i9 = ground_truth;
            auto i6 = minus( i9, i5 );

            if (1)
            {
                auto i6_ = s.run( i6 );
                //std::cerr << "i6:\n" << i6_ << std::endl;
                auto i5_ = s.run( i5 );
                //std::cerr << "i5:\n" << i5_ << std::endl;
                auto i2_ = s.run( i2 );
                //std::cerr << "i2:\n" << i2_ << std::endl;
                auto xt = _x;
                xt.reshape( {dims, 1} );

                auto w_g_ = -2.0 * xt * elementwise_product(i6_, elementwise_product( i5_, ( 1.0 - i5_ ) ) );
                //std::cerr << "Expected Gradient:\n" << w_g_ << std::endl;


                auto w_g_diff = w_gradient - w_g_;

                for ( auto idx : range( w_g_diff.size() ) )
                {
                    debug_print( "DEBUG at idx ", idx,  " w_gradient is ", w_gradient[idx], " and w_g_ is ", w_g_[idx] );
                    REQUIRE( std::abs( w_g_diff[idx] ) < 1.0e-7 );
                }
            }

            /*
            for ( auto idx : range( w_gradient.size() ) )
            {
                debug_print( "DEBUG at idx ", idx,  " w_gradient is ", w_gradient[idx] );
            }
            */


            s.run( optimizer );
            // check the updated weights
            auto w_data_new = (*(W.data_)).deep_copy();
            auto w_diff = w_data - learning_rate * w_gradient - w_data_new;
            for ( auto idx : range( w_diff.size() ) )
                REQUIRE( std::abs( w_diff[idx] ) < 1.0e-7 );

            auto b_data_new = (*(b.data_)).deep_copy();
            auto b_diff = b_data - learning_rate * b_gradient - b_data_new;
            for ( auto idx : range( b_diff.size() ) )
                REQUIRE( std::abs( b_diff[idx] ) < 1.0e-7 );

        }
    }
}

