#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "../include/ceras.hpp"
#include <cmath>

TEST_CASE("softmax", "[softmax]")
{
    using namespace ceras;

    unsigned long const tests = 1024;
    unsigned long const upper_dims = 7;

    std::vector<unsigned long> vdims;
    for ( auto idx : range( upper_dims ) )
        vdims.push_back( idx+1 );

    for ( [[maybe_unused]] auto _ : range( tests ) )
    {
        for ( unsigned long dims_ = 0UL; dims_ != upper_dims; ++dims_ )
        {
            unsigned long dims = dims_ + 1;

            auto _x = random<double>( {1, dims} );
            auto tw = random<double>( {dims, dims} );
            auto tb = random<double>( {1, dims} );

            auto x = place_holder<double>{};
            auto c = place_holder<double>{};
            auto W = variable<double>{ tw };
            auto b = variable<double>{ tb };

            auto p = softmax( x * W + b );

            session<double> s;
            s.bind( x, _x );
            auto result = s.run( p );

            auto result_ = _x * tw + tb;
            result_ -= amax( result_ );
            result_.map( []( double& x ){ x = std::exp(x); } );
            result_ /= sum( result_ );

            auto df = result - result_;

            for ( auto idx : range( df.size() ) )
                REQUIRE( std::abs( df[idx] ) < 1.0e-7 );
        }
    }
}

