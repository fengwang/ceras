#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "../include/ceras.hpp"
#include "../include/tensor.hpp"
#include "../include/backend/cuda.hpp"
#include <cmath>

using namespace ceras;

template< typename T >
auto test( unsigned long row, unsigned long col )
{
    auto tsor_a = random<T>( {row, col} );
    auto tsor_b = zeros<T>( {row, col} );

    T* device_ptr = allocate<T>( row*col );

    host_to_device_n( tsor_a.data(), row*col, device_ptr );
    device_to_host_n( device_ptr, row*col, tsor_b.data() );

    deallocate( device_ptr );

    tsor_a -= tsor_b;
    return tsor_a;
}


TEST_CASE("cuda_memcpy", "[cuda_memcpy]")
{

    //unsigned long const tests = 1024;
    unsigned long const tests = 2;
    unsigned long const upper_dims = 201;

    for ( auto r : range( upper_dims ) )
    {
        auto R = r;
        for ( auto c : range( upper_dims ) )
        {
            auto C = c;

            for ( [[maybe_unused]]auto t : range( tests ) )
            {
                {
                    auto diff = test<float>( R, C );
                    for ( auto v: diff )
                    {
                        REQUIRE( v < 1.0e-10 );
                    }
                }
                {
                    auto diff = test<double>( R, C );
                    for ( auto v: diff )
                    {
                        REQUIRE( v < 1.0e-10 );
                    }
                }
            }
        }
    }
}

