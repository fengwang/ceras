#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "../include/ceras.hpp"
#include "../include/tensor.hpp"
#include <cmath>

using namespace ceras;

#if 0

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
    //unsigned long const tests = 1024;
    unsigned long const tests = 2;
    unsigned long const upper_dims = 11;


    for ( auto r : range(1UL, upper_dims ) )
    {
        auto R = r;
        for ( auto c : range(1UL, upper_dims ) )
        {
            auto C = c;
            for ( auto k : range(1UL, upper_dims ) )
            {
                auto K = k;
                {
                    for ( [[maybe_unused]] auto _: range( tests ) )
                    {
                        {//00
                            auto A = random<float>( {R, K} );
                            auto B = random<float>( {K, C} );
                            auto diff = diff_00( A, B );
                            auto rm = reduce_sum( diff ) / (R*C);
                            auto mn = rm[0];
                            REQUIRE( mn < 1.0e-5 );
                        }
                        {//00
                            auto A = random<double>( {R, K} );
                            auto B = random<double>( {K, C} );
                            auto diff = diff_00( A, B );
                            auto rm = reduce_sum( diff ) / (R*C);
                            auto mn = rm[0];
                            REQUIRE( mn < 1.0e-5 );
                        }
                    }
                }
            }
        }
    }
}

#endif

#if 0

// A[m][n], B[k][n]
template< Tensor Tsor >
auto diff_01( Tsor const& A, Tsor const& B )
{
    typedef typename Tsor::value_type value_type;

    unsigned long m = *(A.shape().begin());
    unsigned long n = *(A.shape().begin()+1);
    unsigned long k = *(B.shape().begin());

    Tsor cuda_ab{{m, k}};
    //  A or A' is [m x n], B or B' is [n x k] and C is [m x k]
    //oid gemm( T const* A, bool a_transposed, T const* B, bool b_transposed, std::size_t m, std::size_t n, std::size_t k, T* C )
    gemm( A.data(), false, B.data(), true, m, n, k, cuda_ab.data() );

    Tsor ab{ {m, n} };
    std::fill( ab.begin(), ab.end(), value_type{0} );

    view_2d<value_type> a{ A.data(), m, n };
    view_2d<value_type> b{ B.data(), k, n };
    view_2d<value_type> v{ ab.data(), m, k };

    for ( auto r : range( m ) )
    {
        for ( auto c : range( n ) )
        {
            for ( auto x : range( k ) )
            {
                v[r][x] += a[r][c] * b[x][c];
            }
        }
    }

    return ab - cuda_ab;
}


TEST_CASE("tensor_mm_01", "[tensor_mm_01]")
{
    {
        auto [m, n, k] = std::make_tuple(3UL, 5UL, 7UL);
        tensor<double> a = random<double>( {m, n} );
        tensor<double> b = random<double>( {k, n} );
        tensor<double> c = random<double>( {m, k} );
        gemm( a.data(), false, b.data(), true, m, n, k, c.data() );
        std::cout << a << std::endl;
        std::cout << b << std::endl;
        std::cout << c << std::endl;
    }


    //unsigned long const tests = 1024;
    unsigned long const tests = 2;
    unsigned long const upper_dims = 31;


    for ( auto r : range(1UL, upper_dims ) )
    {
        auto R = r+3;
        for ( auto c : range(1UL, upper_dims ) )
        {
            auto C = c+5;
            for ( auto k : range(1UL, upper_dims ) )
            {
                auto K = k+7;
                {
                    for ( [[maybe_unused]] auto _: range( tests ) )
                    {
                        {//00
                            auto A = random<float>( {R, K} );
                            auto B = random<float>( {C, K} );
                            auto diff = diff_01( A, B );
                            auto rm = reduce_sum( diff ) / (R*C);
                            auto mn = rm[0];
                            REQUIRE( mn < 1.0e-5 );
                        }
                        {//00
                            auto A = random<double>( {R, K} );
                            auto B = random<double>( {C, K} );
                            auto diff = diff_01( A, B );
                            auto rm = reduce_sum( diff ) / (R*C);
                            auto mn = rm[0];
                            REQUIRE( mn < 1.0e-5 );
                        }
                    }
                }
            }
        }
    }
}

#endif

//void gemm( T const* A, bool a_transposed, T const* B, bool b_transposed, std::size_t m, std::size_t n, std::size_t k, T* C )
TEST_CASE("tensor_gemm", "[tensor_gemm]")
{

    //unsigned long const tests = 1024;
    unsigned long const tests = 2;
    unsigned long const upper_dims = 31;


    for ( auto m : range(1UL, upper_dims ) )
    {
        for ( auto n : range(1UL, upper_dims ) )
        {
            for ( auto k : range(1UL, upper_dims ) )
            {
                {
                    for ( [[maybe_unused]] auto _: range( tests ) )
                    {
                        auto A = random<double>( {m*n,} );
                        auto B = random<double>( {n*k,} );
                        auto C_cpu = zeros<double>( {m*k,} );
                        auto C_gpu = zeros<double>( {m*k,} );

                        if ( 1 )
                        {
                            gemm( A.data(), false, B.data(), false, m, n, k, C_gpu.data() );
                            gemm_cpu( A.data(), false, B.data(), false, m, n, k, C_cpu.data() );
                            auto diff = C_gpu - C_cpu;
                            for ( auto x : diff )
                                REQUIRE( std::abs(x) < 1.0e-5 );
                        }

                        if ( 1 )
                        {
                            gemm( A.data(), true, B.data(), false, m, n, k, C_gpu.data() );
                            gemm_cpu( A.data(), true, B.data(), false, m, n, k, C_cpu.data() );
                            auto diff = C_gpu - C_cpu;
                            for ( auto x : diff )
                                REQUIRE( std::abs(x) < 1.0e-5 );
                        }

                        if ( 1 )
                        {
                            gemm( A.data(), true, B.data(), true, m, n, k, C_gpu.data() );
                            gemm_cpu( A.data(), true, B.data(), true, m, n, k, C_cpu.data() );
                            auto diff = C_gpu - C_cpu;
                            for ( auto x : diff )
                                REQUIRE( std::abs(x) < 1.0e-5 );
                        }

                        if ( 1 )
                        {
                            gemm( A.data(), false, B.data(), true, m, n, k, C_gpu.data() );
                            gemm_cpu( A.data(), false, B.data(), true, m, n, k, C_cpu.data() );
                            auto diff = C_gpu - C_cpu;
                            for ( auto x : diff )
                                REQUIRE( std::abs(x) < 1.0e-5 );
                        }
                    }
                }
            }
        }
    }
}



