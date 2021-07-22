#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "../include/ceras.hpp"
#include "../include/utils/buffered_allocator.hpp"
#include <cmath>

TEST_CASE("buffered_tensor_add", "[buffered_tensor_add]")
{
    using namespace ceras;

    unsigned long const tests = 1024;
    unsigned long const upper_dims = 7;

    std::vector<unsigned long> vdims;
    for ( auto idx : range( upper_dims ) )
        vdims.push_back( idx+1 );

    typedef buffered_allocator<double, 32> allocator_type;

    for ( auto _ : range( tests ) )
    {
        for ( unsigned long dims_ = 0UL; dims_ != upper_dims; ++dims_ )
        {
            unsigned long dims = dims_ + 1;

            auto a = random<double, allocator_type>( {dims, dims} );
            auto b = random<double, allocator_type>( {1, dims} );
            auto c = random<double, allocator_type>( {dims, dims} );

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

void print( auto vec )
{
    for ( auto i = 0UL; i != vec.size(); ++i )
        std::cout << vec[i] << " ";
    std::cout << std::endl;
}


TEST_CASE("vec", "[vec]")
{
    using namespace ceras;

    std::vector<int, buffered_allocator<int,50>> vec;
    unsigned long const N = 60;

    //increase
    for ( auto i = 0UL; i != N; ++i )
    {
        vec.push_back( i );
        //vec.resize( i+1 );
        print( vec );
    }

    //decrease
    for ( auto i = 0UL; i != N; ++i )
    {
        vec.resize( N - i - 1 );
        print( vec );
    }

}

std::string create_string( unsigned long N )
{
    std::string str;
    str.resize( N );
    std::string dat = std::to_string( N );
    std::fill( str.begin(), str.end(), dat[0] );
    return str;
}


TEST_CASE("vstring", "[vstring]")
{
    using namespace ceras;

    std::vector<std::string, buffered_allocator<std::string,1000000>> vec;
    unsigned long const N = 25;

    //increase
    for ( auto i = 0UL; i != N; ++i )
    {
        vec.push_back( create_string(i) );
        //vec.resize( i+1 );
        print( vec );
    }

    //decrease
    for ( auto i = 0UL; i != N; ++i )
    {
        vec.resize( N - i - 1 );
        print( vec );
    }

}



