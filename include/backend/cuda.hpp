#ifndef CUDA_HPP_INCLUDED_DOPSIHASLKJASLKJSLKJSDALKSJDLSAKJDFFFDDDDDD
#define CUDA_HPP_INCLUDED_DOPSIHASLKJASLKJSLKJSDALKSJDLSAKJDFFFDDDDDD

#include "../includes.hpp"
#include "../config.hpp"
#include "../utils/singleton.hpp"
#include "../utils/debug.hpp"
#include "../utils/timer.hpp"

extern "C"
{
    typedef enum cudaError_enum
    {
        CUDA_SUCCESS = 0,
        CUDA_ERROR_INVALID_VALUE = 1,
        CUDA_ERROR_OUT_OF_MEMORY = 2,
        CUDA_ERROR_NOT_INITIALIZED = 3,
        CUDA_ERROR_DEINITIALIZED = 4,
        CUDA_ERROR_PROFILER_DISABLED = 5,
        CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
        CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
        CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
        CUDA_ERROR_STUB_LIBRARY = 34,
        CUDA_ERROR_NO_DEVICE = 100,
        CUDA_ERROR_INVALID_DEVICE = 101,
        CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
        CUDA_ERROR_INVALID_IMAGE = 200,
        CUDA_ERROR_INVALID_CONTEXT = 201,
        CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
        CUDA_ERROR_MAP_FAILED = 205,
        CUDA_ERROR_UNMAP_FAILED = 206,
        CUDA_ERROR_ARRAY_IS_MAPPED = 207,
        CUDA_ERROR_ALREADY_MAPPED = 208,
        CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
        CUDA_ERROR_ALREADY_ACQUIRED = 210,
        CUDA_ERROR_NOT_MAPPED = 211,
        CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
        CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
        CUDA_ERROR_ECC_UNCORRECTABLE = 214,
        CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
        CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
        CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
        CUDA_ERROR_INVALID_PTX = 218,
        CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
        CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
        CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
        CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
        CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
        CUDA_ERROR_INVALID_SOURCE = 300,
        CUDA_ERROR_FILE_NOT_FOUND = 301,
        CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
        CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
        CUDA_ERROR_OPERATING_SYSTEM = 304,
        CUDA_ERROR_INVALID_HANDLE = 400,
        CUDA_ERROR_ILLEGAL_STATE = 401,
        CUDA_ERROR_NOT_FOUND = 500,
        CUDA_ERROR_NOT_READY = 600,
        CUDA_ERROR_ILLEGAL_ADDRESS = 700,
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
        CUDA_ERROR_LAUNCH_TIMEOUT = 702,
        CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
        CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
        CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
        CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
        CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
        CUDA_ERROR_ASSERT = 710,
        CUDA_ERROR_TOO_MANY_PEERS = 711,
        CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
        CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
        CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
        CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
        CUDA_ERROR_MISALIGNED_ADDRESS = 716,
        CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
        CUDA_ERROR_INVALID_PC = 718,
        CUDA_ERROR_LAUNCH_FAILED = 719,
        CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
        CUDA_ERROR_NOT_PERMITTED = 800,
        CUDA_ERROR_NOT_SUPPORTED = 801,
        CUDA_ERROR_SYSTEM_NOT_READY = 802,
        CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
        CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
        CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
        CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
        CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
        CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
        CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
        CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
        CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
        CUDA_ERROR_CAPTURED_EVENT = 907,
        CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
        CUDA_ERROR_TIMEOUT = 909,
        CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
        CUDA_ERROR_UNKNOWN = 999
    } CUresult;

    typedef enum
    {
        CUBLAS_STATUS_SUCCESS = 0,
        CUBLAS_STATUS_NOT_INITIALIZED = 1,
        CUBLAS_STATUS_ALLOC_FAILED = 3,
        CUBLAS_STATUS_INVALID_VALUE = 7,
        CUBLAS_STATUS_ARCH_MISMATCH = 8,
        CUBLAS_STATUS_MAPPING_ERROR = 11,
        CUBLAS_STATUS_EXECUTION_FAILED = 13,
        CUBLAS_STATUS_INTERNAL_ERROR = 14,
        CUBLAS_STATUS_NOT_SUPPORTED = 15,
        CUBLAS_STATUS_LICENSE_ERROR = 16
    } cublasStatus_t;

    const char* cudaGetErrorString ( CUresult error );

    int printf( const char* __restrict, ... );

    CUresult cudaGetLastError ( void );
}//extern "C"

#ifdef cuda_assert
#undef cuda_assert
#endif
#define cuda_assert( fn ) do { \
  CUresult status = (fn); \
  if ( CUDA_SUCCESS != status ) { \
    char const* errstr = cudaGetErrorString(status); \
    printf("CUDA Driver Failure (line %d of file %s):\n\t%s returned 0x%x (%s)\n", __LINE__, __FILE__, #fn, status, errstr); \
    exit(EXIT_FAILURE); \
  } \
} while (0)


#ifdef cublas_assert
#undef cublas_assert
#endif

inline void cuda_no_error_so_far()
{
    cuda_assert( cudaGetLastError() );
}

struct cublas_result_assert
{
    void operator()( const cublasStatus_t& result, const char* const file, const unsigned long line ) const
    {
        if ( CUBLAS_STATUS_SUCCESS == result ) return;

        report_error( result, file, line );
    }

    void report_error( const cublasStatus_t& result, const char* const file, const unsigned long line ) const
    {
        printf( "%s:%lu: cuda runtime error occured:\n[[ERROR]]: %s\n", file, line, error_msg( result ) );
        abort();
    }

    const char* error_msg( const cublasStatus_t& result ) const
    {
        if ( result == CUBLAS_STATUS_NOT_INITIALIZED )
            return "The CUBLAS library was not initialized.  This is usually caused by the lack of a prior cublasCreate() call, an error in the CUDA Runtime API called by the CUCUBLAS routine, or an error in the hardware setup.";

        if ( result == CUBLAS_STATUS_ALLOC_FAILED )
            return "Resource allocation failed inside the CUBLAS library. This is usually caused by a cudaMalloc() failure.";

        if ( result == CUBLAS_STATUS_INVALID_VALUE )
            return "An unsupported value or parameter was passed to the function (a negative vector size, for example).";

        if ( result == CUBLAS_STATUS_ARCH_MISMATCH )
            return "The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.";

        if ( result == CUBLAS_STATUS_MAPPING_ERROR )
            return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.";

        if ( result == CUBLAS_STATUS_EXECUTION_FAILED )
            return "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.";

        if ( result == CUBLAS_STATUS_INTERNAL_ERROR )
            return "An internal CUBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure.";

        if ( result == CUBLAS_STATUS_NOT_SUPPORTED )
            return "Not supported.";

        if ( result == CUBLAS_STATUS_LICENSE_ERROR )
            return "License error.";

        return "an unknown internal error has occurred.";
    }
};//struct cublas_result_assert

#define cublas_assert(result) cublas_result_assert()(result, __FILE__, __LINE__)

extern "C"
{
    enum cudaMemcpyKind
    {
        cudaMemcpyHostToHost = 0,
        cudaMemcpyHostToDevice = 1,
        cudaMemcpyDeviceToHost = 2,
        cudaMemcpyDeviceToDevice = 3,
        cudaMemcpyDefault = 4
    };

    CUresult cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind );
}

namespace ceras
{

    template<typename Type>
    void host_to_device_n( const Type* host_begin, std::size_t length, Type* device_begin )
    {
        cuda_assert( cudaMemcpy( reinterpret_cast<void*>( device_begin ), reinterpret_cast<const void*>( host_begin ), length * sizeof( Type ), cudaMemcpyHostToDevice ) );
    }

    template<typename Type>
    void host_to_device( const Type* host_begin, const Type* host_end, Type* device_begin )
    {
        std::size_t length = std::distance( host_begin, host_end );
        host_to_device_n( host_begin, length, device_begin );
    }

    template<typename Type>
    void device_to_host_n( const Type* device_begin, std::size_t length, Type* host_begin )
    {
        cuda_assert( cudaMemcpy( reinterpret_cast<void*>( host_begin ), reinterpret_cast<const void*>( device_begin ), length * sizeof( Type ), cudaMemcpyDeviceToHost ) );
    }

    template<typename Type>
    void device_to_host( const Type* device_begin, const Type* device_end, Type* host_begin )
    {
        std::size_t length = std::distance( device_begin, device_end );
        device_to_host_n( device_begin, length, host_begin );
    }

}//namespace ceras

extern "C"
{
    CUresult cudaMalloc ( void** devPtr, size_t size );
    CUresult cudaMallocHost ( void** ptr, size_t size );
    CUresult cudaFree ( void* devPtr );
    CUresult cudaFreeHost ( void* devPtr );
}

namespace ceras
{
    template< typename T >
    T* allocate( unsigned long n )
    {
        T* ans;
        cudaMalloc( reinterpret_cast<void**>( &ans ), n * sizeof( T ) );
        return ans;
    }

    template< typename T >
    void deallocate( T* ptr )
    {
        cudaFree( reinterpret_cast<void*>( ptr ) );
    }

    template< typename T >
    T* allocate_host( unsigned long n )
    {
        T* ans;
        cudaMallocHost( reinterpret_cast<void**>( &ans ), n * sizeof( T ) );
        return ans;
    }

    template< typename T >
    void deallocate_host( T* ptr )
    {
        cudaFreeHost( reinterpret_cast<void*>( ptr ) );
    }

    struct cuda_memory_cache
    {
        void* data_;
        std::size_t size_;

        cuda_memory_cache(): data_{ nullptr }, size_{ 0 } {}

        template< typename T >
        void reserve( std::size_t length )
        {
            std::size_t new_size = length * sizeof( T ); // size in bytes
            if ( size_ >= new_size )
                return;

            new_size = (1 + ((new_size+size_) >> 20)) << 20; // at least 1 MB
            std::size_t new_length = new_size / sizeof(T);

            if ( size_ > 0 )
                deallocate( data_ );

            data_ = allocate<T>( new_length );
            size_ = new_size;
        }

        template< typename T >
        T* data()
        {
            return reinterpret_cast<T*>( data_ );
        }

        ~cuda_memory_cache()
        {
            if ( size_ > 0 )
                deallocate( data_ );
            size_ = 0;
            data_ = nullptr;
        }

    };
}//namespace ceras

extern "C"
{

    CUresult cudaSetDevice( int );
    CUresult cudaGetDevice( int* );
    CUresult cudaDeviceSynchronize ( void );

    struct cublasContext;
    typedef struct cublasContext* cublasHandle_t;

    typedef enum
    {
        CUBLAS_OP_N=0,
        CUBLAS_OP_T=1,
        CUBLAS_OP_C=2,
        CUBLAS_OP_HERMITAN=2,
        CUBLAS_OP_CONJG=3
    } cublasOperation_t;

    cublasStatus_t cublasCreate_v2 ( cublasHandle_t* handle );
    cublasStatus_t cublasDestroy_v2 ( cublasHandle_t handle );
    cublasStatus_t cublasSgemm_v2( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                   int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc );
    cublasStatus_t cublasDgemm_v2( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                   int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc );
}

namespace ceras
{

    struct cublas_handle
    {
        cublasHandle_t handle_;
        cublas_handle()
        {
            int current_id;
            cuda_assert( cudaGetDevice(&current_id) );
            if ( current_id != visible_device ) // visible_device is defined in config.hpp
                cuda_assert( cudaSetDevice( visible_device ) );
            cublas_assert( cublasCreate_v2( &handle_ ) );
        }

        ~cublas_handle()
        {
            if constexpr( cuda_mode )
                cublas_assert( cublasDestroy_v2( handle_ ) );
        }
    };//cublas_handle_delegate

    // C <= A * B
    // where A or A' is [m x n], B or B' is [n x k] and C is [m x k]
    template< typename T > requires std::floating_point<T>
    void cuda_gemm( T const* A, bool a_transposed, T const* B, bool b_transposed, std::size_t m, std::size_t n, std::size_t k, T* C )
    {
        cublas_handle& handle = singleton<cublas_handle>::instance();
        cublasHandle_t hd = handle.handle_;

        T alpha = 1.0;
        T beta = 0.0;

        cuda_memory_cache& cache = singleton<cuda_memory_cache>::instance();
        cache.reserve<T>( m*n + n*k + m*k ); // total memory
        T* a = cache.data<T>();
        host_to_device_n( A, m*n, a );
        T* b = a + m*n;
        host_to_device_n( B, n*k, b );
        T* c = b + n*k;
        // not copying this
        //host_to_device_n( C, m*k, c );

        T* result_ptr = c;
        T* first_ptr = b;
        T* second_ptr = a;
        int row_of_c = m;
        int col_of_c = k;
        int common_dimension = n;

        int ld_of_first_ptr = b_transposed ? n : k;
        int ld_of_second_ptr = a_transposed ? m : n;
        int ld_of_result_ptr = k;

        cublasOperation_t first_transposed = b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t second_transposed = a_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;

        if constexpr( std::is_same_v<T, float> )
        {
            cublas_assert( cublasSgemm_v2( hd, first_transposed, second_transposed, col_of_c, row_of_c, common_dimension, &alpha, first_ptr, ld_of_first_ptr, second_ptr, ld_of_second_ptr, &beta, result_ptr, ld_of_result_ptr ) );
        }
        else if constexpr( std::is_same_v<T, double> )
        {
            cublas_assert( cublasDgemm_v2( hd, first_transposed, second_transposed, col_of_c, row_of_c, common_dimension, &alpha, first_ptr, ld_of_first_ptr, second_ptr, ld_of_second_ptr, &beta, result_ptr, ld_of_result_ptr ) );
        }
        else
        {
            better_assert( false, "gemm only supports float and double!" );
        }

        // device -> host
        device_to_host_n( c, m*k,  C );
    }

}//namespace ceras

#endif//HLUDCUYMXCGREGWXHAVFNPYCNJLENNKSIOVPKKRXDUBYCAEMYBOKVKWODFGLJOUYYLQEDWSTB

