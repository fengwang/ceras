#ifndef CBLAS_HPP_INCLUDED_SDOASDFAPOSJIASLDFKJASLDFKJALSDFKJDASLKJDFSFDFFFFSKL
#define CBLAS_HPP_INCLUDED_SDOASDFAPOSJIASLDFKJASLDFKJALSDFKJDASLKJDFSFDFFFFSKL

#include "../includes.hpp"
#include "../config.hpp"
#include "../utils/debug.hpp"
#include "../utils/better_assert.hpp"

extern "C"
{

typedef enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
typedef int CBLAS_INDEX;

void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N,
                 const CBLAS_INDEX K, const float alpha, const float *A,
                 const CBLAS_INDEX lda, const float *B, const CBLAS_INDEX ldb,
                 const float beta, float *C, const CBLAS_INDEX ldc);

void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N,
                 const CBLAS_INDEX K, const double alpha, const double *A,
                 const CBLAS_INDEX lda, const double *B, const CBLAS_INDEX ldb,
                 const double beta, double *C, const CBLAS_INDEX ldc);
}


namespace ceras
{

    // C <= A * B
    // where A or A' is [m x n], B or B' is [n x k] and C is [m x k]
    template< typename T > requires std::floating_point<T>
    void cblas_gemm( T const* A, bool a_transposed, T const* B, bool b_transposed, std::size_t m, std::size_t n, std::size_t k, T* C )
    {
        if constexpr( cblas_mode )
        {
            T const alpha = 1.0;
            T const beta = 0.0;

            T* result_ptr = C;
            T const * first_ptr = B;
            T const * second_ptr = A;
            int const row_of_c = m;
            int const col_of_c = k;
            int const common_dimension = n;

            int const ld_of_first_ptr = b_transposed ? n : k;
            int const ld_of_second_ptr = a_transposed ? m : n;
            int const ld_of_result_ptr = k;

            CBLAS_TRANSPOSE const first_transposed = b_transposed ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE const second_transposed = a_transposed ? CblasTrans : CblasNoTrans;

            if constexpr( std::is_same_v<T, float> )
            {
                cblas_sgemm( CblasColMajor, first_transposed, second_transposed, col_of_c, row_of_c, common_dimension, alpha, first_ptr, ld_of_first_ptr, second_ptr, ld_of_second_ptr, beta, result_ptr, ld_of_result_ptr );
            }
            else if constexpr( std::is_same_v<T, double> )
            {
                cblas_dgemm( CblasColMajor, first_transposed, second_transposed, col_of_c, row_of_c, common_dimension, alpha, first_ptr, ld_of_first_ptr, second_ptr, ld_of_second_ptr, beta, result_ptr, ld_of_result_ptr );
            }
            else
            {
                better_assert( false, "Error: blas_gemm only supports float and double!" );
            }
        }
        else
        {
            better_assert( false, "Error: calling blas_gemm but blas_mode is not enabled!" );
        }
    }

}//namespace ceras

#endif//CBLAS_HPP_INCLUDED_SDOASDFAPOSJIASLDFKJASLDFKJALSDFKJDASLKJDFSFDFFFFSKL

