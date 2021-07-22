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

    for ( [[maybe_unused]] auto _ : range( tests ) )
    {
        for ( unsigned long dims_ = 0UL; dims_ != upper_dims; ++dims_ )
        {
            unsigned long dims = dims_ + 1;

            auto a = random<double>( {dims, dims} );
            save_tensor( "./iotest.txt", a );
            auto b = load_tensor<double>( "./iotest.txt" ).reshape({dims, dims});


            auto diff = a - b;
            auto nm = norm( a - b );

            REQUIRE( nm < 1.0e-5 );
        }
    }
}

