#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "../include/ceras.hpp"
#include "../include/tensor.hpp"
#include <cmath>

using namespace ceras;


// gemm( view_2d<T> const& x, view_2d<T> const& y, view_2d<T>& ans )
template< Tensor Tsor >
auto diff_00( Tsor const& A, Tsor const& B )
{
    typedef typename Tsor::value_type value_type;
    auto cuda_ab = A * B;

    unsigned long m = *(A.shape().begin());
    unsigned long k = *(A.shape().begin()+1);
    unsigned long n = *(B.shape().rbegin());
    Tsor ab{ {m, n} };
    std::fill( ab.begin(), ab.end(), value_type{0} );

    view_2d<value_type> a{ A.data(), m, k };
    view_2d<value_type> b{ B.data(), k, n };
    view_2d<value_type> v{ ab.data(), m, n };

    for ( auto r : range( m ) )
    {
        for ( auto c : range( n ) )
        {
            for ( auto x : range( k ) )
            {
                v[r][c] += a[r][x] * b[x][c];
            }
        }
    }

    return ab - cuda_ab;
}


TEST_CASE("tensor_mm_00", "[tensor_mm_00]")
{

    {
        auto a = random<float>( {1, 2} );
        auto b = random<float>( {2, 7} );
        auto c = a * b;
        std::cout << "a:\n" << a << std::endl;
        std::cout << "b:\n" << b << std::endl;
        std::cout << "c:\n" << c << std::endl;
    }



    //unsigned long const tests = 1024;
    unsigned long const tests = 1;
    unsigned long const upper_dims = 7;


    for ( auto r : range( upper_dims ) )
    {
        auto R = r + 11;
        for ( auto c : range( upper_dims ) )
        {
            auto C = c + 21;
            for ( auto k : range( upper_dims ) )
            {
                auto K = k + 31;
                {
                    for ( [[maybe_unused]] auto _: range( tests ) )
                    {
                        {//00
                            auto A = random<float>( {R, K} );
                            auto B = random<float>( {K, C} );
                            auto diff = diff_00( A, B );
                            auto rm = reduce_sum( diff );
                            auto mn = rm[0];
                            REQUIRE( mn < 1.0e-5 );
                        }
                    }
                }
            }
        }
    }

}

