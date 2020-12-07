#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "../include/ceras.hpp"
#include <cmath>

TEST_CASE("tensor_add", "[tensor_add]")
{
    using namespace ceras;

    unsigned long const tests = 1024;
    unsigned long const upper_dims = 7;

    std::vector<unsigned long> vdims;
    for ( auto idx : range( upper_dims ) )
        vdims.push_back( idx+1 );

    for ( auto _ : range( tests ) )
    {
        for ( unsigned long dims_ = 0UL; dims_ != upper_dims; ++dims_ )
        {
            unsigned long dims = dims_ + 1;

            auto a = numeric::random<double>( {dims, dims} );
            auto b = numeric::random<double>( {1, dims} );
            auto c = numeric::random<double>( {dims, dims} );

            auto ab = a+b;
            for ( auto r : range( dims ) )
                for ( auto c : range( dims ) )
                {
                    auto diff = ab[r*dims+c] - (a[r*dims+c] + b[c]);
                    REQUIRE( std::abs( diff ) < 1.0e-7 );
                }

            auto ba = b+a;
            auto abba_diff = ab - ba;
            for ( auto idx : range( abba_diff.size() ) )
                REQUIRE( std::abs( abba_diff[idx] ) < 1.0e-7 );

            auto ac = a+c;
            for ( auto r : range( dims ) )
                for ( auto c_ : range( dims ) )
                {
                    auto diff = ac[r*dims+c_] - (a[r*dims+c_] + c[r*dims+c_]);
                    REQUIRE( std::abs( diff ) < 1.0e-7 );
                }
        }
    }
}

