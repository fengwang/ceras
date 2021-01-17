#ifndef CUDA_HPP_INCLUDED_DOPSIHASLKJASLKJSLKJSDALKSJDLSAKJDFFFDDDDDD
#define CUDA_HPP_INCLUDED_DOPSIHASLKJASLKJSLKJSDALKSJDLSAKJDFFFDDDDDD

#include "../includes.hpp"
#include "../config.hpp"
#include "../utils/singleton.hpp"
#include "../utils/debug.hpp"

extern "C"
{

    /**
        Error codes
    */
    typedef enum cudaError_enum
    {
        /**
            The API call returned with no errors. In the case of query calls, this
            also means that the operation being queried is complete (see
            ::cuEventQuery() and ::cuStreamQuery()).
        */
        CUDA_SUCCESS                              = 0,

        /**
            This indicates that one or more of the parameters passed to the API call
            is not within an acceptable range of values.
        */
        CUDA_ERROR_INVALID_VALUE                  = 1,

        /**
            The API call failed because it was unable to allocate enough memory to
            perform the requested operation.
        */
        CUDA_ERROR_OUT_OF_MEMORY                  = 2,

        /**
            This indicates that the CUDA driver has not been initialized with
            ::cuInit() or that initialization has failed.
        */
        CUDA_ERROR_NOT_INITIALIZED                = 3,

        /**
            This indicates that the CUDA driver is in the process of shutting down.
        */
        CUDA_ERROR_DEINITIALIZED                  = 4,

        /**
            This indicates profiler is not initialized for this run. This can
            happen when the application is running with external profiling tools
            like visual profiler.
        */
        CUDA_ERROR_PROFILER_DISABLED              = 5,

        /**
            \deprecated
            This error return is deprecated as of CUDA 5.0. It is no longer an error
            to attempt to enable/disable the profiling via ::cuProfilerStart or
            ::cuProfilerStop without initialization.
        */
        CUDA_ERROR_PROFILER_NOT_INITIALIZED       = 6,

        /**
            \deprecated
            This error return is deprecated as of CUDA 5.0. It is no longer an error
            to call cuProfilerStart() when profiling is already enabled.
        */
        CUDA_ERROR_PROFILER_ALREADY_STARTED       = 7,

        /**
            \deprecated
            This error return is deprecated as of CUDA 5.0. It is no longer an error
            to call cuProfilerStop() when profiling is already disabled.
        */
        CUDA_ERROR_PROFILER_ALREADY_STOPPED       = 8,

        /**
            This indicates that the CUDA driver that the application has loaded is a
            stub library. Applications that run with the stub rather than a real
            driver loaded will result in CUDA API returning this error.
        */
        CUDA_ERROR_STUB_LIBRARY                   = 34,

        /**
            This indicates that no CUDA-capable devices were detected by the installed
            CUDA driver.
        */
        CUDA_ERROR_NO_DEVICE                      = 100,

        /**
            This indicates that the device ordinal supplied by the user does not
            correspond to a valid CUDA device.
        */
        CUDA_ERROR_INVALID_DEVICE                 = 101,

        /**
            This error indicates that the Grid license is not applied.
        */
        CUDA_ERROR_DEVICE_NOT_LICENSED            = 102,

        /**
            This indicates that the device kernel image is invalid. This can also
            indicate an invalid CUDA module.
        */
        CUDA_ERROR_INVALID_IMAGE                  = 200,

        /**
            This most frequently indicates that there is no context bound to the
            current thread. This can also be returned if the context passed to an
            API call is not a valid handle (such as a context that has had
            ::cuCtxDestroy() invoked on it). This can also be returned if a user
            mixes different API versions (i.e. 3010 context with 3020 API calls).
            See ::cuCtxGetApiVersion() for more details.
        */
        CUDA_ERROR_INVALID_CONTEXT                = 201,

        /**
            This indicated that the context being supplied as a parameter to the
            API call was already the active context.
            \deprecated
            This error return is deprecated as of CUDA 3.2. It is no longer an
            error to attempt to push the active context via ::cuCtxPushCurrent().
        */
        CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = 202,

        /**
            This indicates that a map or register operation has failed.
        */
        CUDA_ERROR_MAP_FAILED                     = 205,

        /**
            This indicates that an unmap or unregister operation has failed.
        */
        CUDA_ERROR_UNMAP_FAILED                   = 206,

        /**
            This indicates that the specified array is currently mapped and thus
            cannot be destroyed.
        */
        CUDA_ERROR_ARRAY_IS_MAPPED                = 207,

        /**
            This indicates that the resource is already mapped.
        */
        CUDA_ERROR_ALREADY_MAPPED                 = 208,

        /**
            This indicates that there is no kernel image available that is suitable
            for the device. This can occur when a user specifies code generation
            options for a particular CUDA source file that do not include the
            corresponding device configuration.
        */
        CUDA_ERROR_NO_BINARY_FOR_GPU              = 209,

        /**
            This indicates that a resource has already been acquired.
        */
        CUDA_ERROR_ALREADY_ACQUIRED               = 210,

        /**
            This indicates that a resource is not mapped.
        */
        CUDA_ERROR_NOT_MAPPED                     = 211,

        /**
            This indicates that a mapped resource is not available for access as an
            array.
        */
        CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212,

        /**
            This indicates that a mapped resource is not available for access as a
            pointer.
        */
        CUDA_ERROR_NOT_MAPPED_AS_POINTER          = 213,

        /**
            This indicates that an uncorrectable ECC error was detected during
            execution.
        */
        CUDA_ERROR_ECC_UNCORRECTABLE              = 214,

        /**
            This indicates that the ::CUlimit passed to the API call is not
            supported by the active device.
        */
        CUDA_ERROR_UNSUPPORTED_LIMIT              = 215,

        /**
            This indicates that the ::CUcontext passed to the API call can
            only be bound to a single CPU thread at a time but is already
            bound to a CPU thread.
        */
        CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = 216,

        /**
            This indicates that peer access is not supported across the given
            devices.
        */
        CUDA_ERROR_PEER_ACCESS_UNSUPPORTED        = 217,

        /**
            This indicates that a PTX JIT compilation failed.
        */
        CUDA_ERROR_INVALID_PTX                    = 218,

        /**
            This indicates an error with OpenGL or DirectX context.
        */
        CUDA_ERROR_INVALID_GRAPHICS_CONTEXT       = 219,

        /**
            This indicates that an uncorrectable NVLink error was detected during the
            execution.
        */
        CUDA_ERROR_NVLINK_UNCORRECTABLE           = 220,

        /**
            This indicates that the PTX JIT compiler library was not found.
        */
        CUDA_ERROR_JIT_COMPILER_NOT_FOUND         = 221,

        /**
            This indicates that the provided PTX was compiled with an unsupported toolchain.
        */

        CUDA_ERROR_UNSUPPORTED_PTX_VERSION        = 222,

        /**
            This indicates that the PTX JIT compilation was disabled.
        */
        CUDA_ERROR_JIT_COMPILATION_DISABLED       = 223,

        /**
            This indicates that the device kernel source is invalid.
        */
        CUDA_ERROR_INVALID_SOURCE                 = 300,

        /**
            This indicates that the file specified was not found.
        */
        CUDA_ERROR_FILE_NOT_FOUND                 = 301,

        /**
            This indicates that a link to a shared object failed to resolve.
        */
        CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,

        /**
            This indicates that initialization of a shared object failed.
        */
        CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303,

        /**
            This indicates that an OS call failed.
        */
        CUDA_ERROR_OPERATING_SYSTEM               = 304,

        /**
            This indicates that a resource handle passed to the API call was not
            valid. Resource handles are opaque types like ::CUstream and ::CUevent.
        */
        CUDA_ERROR_INVALID_HANDLE                 = 400,

        /**
            This indicates that a resource required by the API call is not in a
            valid state to perform the requested operation.
        */
        CUDA_ERROR_ILLEGAL_STATE                  = 401,

        /**
            This indicates that a named symbol was not found. Examples of symbols
            are global/constant variable names, texture names, and surface names.
        */
        CUDA_ERROR_NOT_FOUND                      = 500,

        /**
            This indicates that asynchronous operations issued previously have not
            completed yet. This result is not actually an error, but must be indicated
            differently than ::CUDA_SUCCESS (which indicates completion). Calls that
            may return this value include ::cuEventQuery() and ::cuStreamQuery().
        */
        CUDA_ERROR_NOT_READY                      = 600,

        /**
            While executing a kernel, the device encountered a
            load or store instruction on an invalid memory address.
            This leaves the process in an inconsistent state and any further CUDA work
            will return the same error. To continue using CUDA, the process must be terminated
            and relaunched.
        */
        CUDA_ERROR_ILLEGAL_ADDRESS                = 700,

        /**
            This indicates that a launch did not occur because it did not have
            appropriate resources. This error usually indicates that the user has
            attempted to pass too many arguments to the device kernel, or the
            kernel launch specifies too many threads for the kernel's register
            count. Passing arguments of the wrong size (i.e. a 64-bit pointer
            when a 32-bit int is expected) is equivalent to passing too many
            arguments and can also result in this error.
        */
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701,

        /**
            This indicates that the device kernel took too long to execute. This can
            only occur if timeouts are enabled - see the device attribute
            ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information.
            This leaves the process in an inconsistent state and any further CUDA work
            will return the same error. To continue using CUDA, the process must be terminated
            and relaunched.
        */
        CUDA_ERROR_LAUNCH_TIMEOUT                 = 702,

        /**
            This error indicates a kernel launch that uses an incompatible texturing
            mode.
        */
        CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,

        /**
            This error indicates that a call to ::cuCtxEnablePeerAccess() is
            trying to re-enable peer access to a context which has already
            had peer access to it enabled.
        */
        CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    = 704,

        /**
            This error indicates that ::cuCtxDisablePeerAccess() is
            trying to disable peer access which has not been enabled yet
            via ::cuCtxEnablePeerAccess().
        */
        CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        = 705,

        /**
            This error indicates that the primary context for the specified device
            has already been initialized.
        */
        CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708,

        /**
            This error indicates that the context current to the calling thread
            has been destroyed using ::cuCtxDestroy, or is a primary context which
            has not yet been initialized.
        */
        CUDA_ERROR_CONTEXT_IS_DESTROYED           = 709,

        /**
            A device-side assert triggered during kernel execution. The context
            cannot be used anymore, and must be destroyed. All existing device
            memory allocations from this context are invalid and must be
            reconstructed if the program is to continue using CUDA.
        */
        CUDA_ERROR_ASSERT                         = 710,

        /**
            This error indicates that the hardware resources required to enable
            peer access have been exhausted for one or more of the devices
            passed to ::cuCtxEnablePeerAccess().
        */
        CUDA_ERROR_TOO_MANY_PEERS                 = 711,

        /**
            This error indicates that the memory range passed to ::cuMemHostRegister()
            has already been registered.
        */
        CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,

        /**
            This error indicates that the pointer passed to ::cuMemHostUnregister()
            does not correspond to any currently registered memory region.
        */
        CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713,

        /**
            While executing a kernel, the device encountered a stack error.
            This can be due to stack corruption or exceeding the stack size limit.
            This leaves the process in an inconsistent state and any further CUDA work
            will return the same error. To continue using CUDA, the process must be terminated
            and relaunched.
        */
        CUDA_ERROR_HARDWARE_STACK_ERROR           = 714,

        /**
            While executing a kernel, the device encountered an illegal instruction.
            This leaves the process in an inconsistent state and any further CUDA work
            will return the same error. To continue using CUDA, the process must be terminated
            and relaunched.
        */
        CUDA_ERROR_ILLEGAL_INSTRUCTION            = 715,

        /**
            While executing a kernel, the device encountered a load or store instruction
            on a memory address which is not aligned.
            This leaves the process in an inconsistent state and any further CUDA work
            will return the same error. To continue using CUDA, the process must be terminated
            and relaunched.
        */
        CUDA_ERROR_MISALIGNED_ADDRESS             = 716,

        /**
            While executing a kernel, the device encountered an instruction
            which can only operate on memory locations in certain address spaces
            (global, shared, or local), but was supplied a memory address not
            belonging to an allowed address space.
            This leaves the process in an inconsistent state and any further CUDA work
            will return the same error. To continue using CUDA, the process must be terminated
            and relaunched.
        */
        CUDA_ERROR_INVALID_ADDRESS_SPACE          = 717,

        /**
            While executing a kernel, the device program counter wrapped its address space.
            This leaves the process in an inconsistent state and any further CUDA work
            will return the same error. To continue using CUDA, the process must be terminated
            and relaunched.
        */
        CUDA_ERROR_INVALID_PC                     = 718,

        /**
            An exception occurred on the device while executing a kernel. Common
            causes include dereferencing an invalid device pointer and accessing
            out of bounds shared memory. Less common cases can be system specific - more
            information about these cases can be found in the system specific user guide.
            This leaves the process in an inconsistent state and any further CUDA work
            will return the same error. To continue using CUDA, the process must be terminated
            and relaunched.
        */
        CUDA_ERROR_LAUNCH_FAILED                  = 719,

        /**
            This error indicates that the number of blocks launched per grid for a kernel that was
            launched via either ::cuLaunchCooperativeKernel or ::cuLaunchCooperativeKernelMultiDevice
            exceeds the maximum number of blocks as allowed by ::cuOccupancyMaxActiveBlocksPerMultiprocessor
            or ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
            as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
        */
        CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE   = 720,

        /**
            This error indicates that the attempted operation is not permitted.
        */
        CUDA_ERROR_NOT_PERMITTED                  = 800,

        /**
            This error indicates that the attempted operation is not supported
            on the current system or device.
        */
        CUDA_ERROR_NOT_SUPPORTED                  = 801,

        /**
            This error indicates that the system is not yet ready to start any CUDA
            work.  To continue using CUDA, verify the system configuration is in a
            valid state and all required driver daemons are actively running.
            More information about this error can be found in the system specific
            user guide.
        */
        CUDA_ERROR_SYSTEM_NOT_READY               = 802,

        /**
            This error indicates that there is a mismatch between the versions of
            the display driver and the CUDA driver. Refer to the compatibility documentation
            for supported versions.
        */
        CUDA_ERROR_SYSTEM_DRIVER_MISMATCH         = 803,

        /**
            This error indicates that the system was upgraded to run with forward compatibility
            but the visible hardware detected by CUDA does not support this configuration.
            Refer to the compatibility documentation for the supported hardware matrix or ensure
            that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES
            environment variable.
        */
        CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,

        /**
            This error indicates that the operation is not permitted when
            the stream is capturing.
        */
        CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED     = 900,

        /**
            This error indicates that the current capture sequence on the stream
            has been invalidated due to a previous error.
        */
        CUDA_ERROR_STREAM_CAPTURE_INVALIDATED     = 901,

        /**
            This error indicates that the operation would have resulted in a merge
            of two independent capture sequences.
        */
        CUDA_ERROR_STREAM_CAPTURE_MERGE           = 902,

        /**
            This error indicates that the capture was not initiated in this stream.
        */
        CUDA_ERROR_STREAM_CAPTURE_UNMATCHED       = 903,

        /**
            This error indicates that the capture sequence contains a fork that was
            not joined to the primary stream.
        */
        CUDA_ERROR_STREAM_CAPTURE_UNJOINED        = 904,

        /**
            This error indicates that a dependency would have been created which
            crosses the capture sequence boundary. Only implicit in-stream ordering
            dependencies are allowed to cross the boundary.
        */
        CUDA_ERROR_STREAM_CAPTURE_ISOLATION       = 905,

        /**
            This error indicates a disallowed implicit dependency on a current capture
            sequence from cudaStreamLegacy.
        */
        CUDA_ERROR_STREAM_CAPTURE_IMPLICIT        = 906,

        /**
            This error indicates that the operation is not permitted on an event which
            was last recorded in a capturing stream.
        */
        CUDA_ERROR_CAPTURED_EVENT                 = 907,

        /**
            A stream capture sequence not initiated with the ::CU_STREAM_CAPTURE_MODE_RELAXED
            argument to ::cuStreamBeginCapture was passed to ::cuStreamEndCapture in a
            different thread.
        */
        CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD    = 908,

        /**
            This error indicates that the timeout specified for the wait operation has lapsed.
        */
        CUDA_ERROR_TIMEOUT                        = 909,

        /**
            This error indicates that the graph update was not performed because it included
            changes which violated constraints specific to instantiated graph update.
        */
        CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE      = 910,

        /**
            This indicates that an unknown internal error has occurred.
        */
        CUDA_ERROR_UNKNOWN                        = 999
    } CUresult;

    /* CUBLAS status type returns */
    typedef enum
    {
        CUBLAS_STATUS_SUCCESS         = 0,
        CUBLAS_STATUS_NOT_INITIALIZED = 1,
        CUBLAS_STATUS_ALLOC_FAILED    = 3,
        CUBLAS_STATUS_INVALID_VALUE   = 7,
        CUBLAS_STATUS_ARCH_MISMATCH   = 8,
        CUBLAS_STATUS_MAPPING_ERROR   = 11,
        CUBLAS_STATUS_EXECUTION_FAILED = 13,
        CUBLAS_STATUS_INTERNAL_ERROR  = 14,
        CUBLAS_STATUS_NOT_SUPPORTED   = 15,
        CUBLAS_STATUS_LICENSE_ERROR   = 16
    } cublasStatus_t;

    //CUresult cuGetErrorString( CUresult error, const char** pStr );
    const char* cudaGetErrorString ( CUresult error );

    int printf( const char* __restrict, ... );

    CUresult cudaGetLastError ( void );

}


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
        if ( CUBLAS_STATUS_SUCCESS == result )
        {
            return;
        }

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
        {
            return "The CUBLAS library was not initialized.  This is usually caused by the lack of a prior cublasCreate() call, an error in the CUDA Runtime API called by the CUCUBLAS routine, or an error in the hardware setup.";
        }

        if ( result == CUBLAS_STATUS_ALLOC_FAILED )
        {
            return "Resource allocation failed inside the CUBLAS library. This is usually caused by a cudaMalloc() failure.";
        }

        if ( result == CUBLAS_STATUS_INVALID_VALUE )
        {
            return "An unsupported value or parameter was passed to the function (a negative vector size, for example).";
        }

        if ( result == CUBLAS_STATUS_ARCH_MISMATCH )
        {
            return "The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.";
        }

        if ( result == CUBLAS_STATUS_MAPPING_ERROR )
        {
            return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.";
        }

        if ( result == CUBLAS_STATUS_EXECUTION_FAILED )
        {
            return "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.";
        }

        if ( result == CUBLAS_STATUS_INTERNAL_ERROR )
        {
            return "An internal CUBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure.";
        }

        if ( result == CUBLAS_STATUS_NOT_SUPPORTED )
        {
            return "Not supported.";
        }

        if ( result == CUBLAS_STATUS_LICENSE_ERROR )
        {
            return "License error.";
        }

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
    void host_to_device( const Type* hst, Type* dev )
    {
        cuda_assert( cudaMemcpy( reinterpret_cast<void*>( dev ), reinterpret_cast<const void*>( hst ), sizeof( Type ), cudaMemcpyHostToDevice ) );
    }

    template<typename Type>
    void host_to_device_n( const Type* hst_begin, std::size_t length, Type* dev_begin )
    {
        debug_print( "host_to_device_n with from hst_begin: ", hst_begin, ", length: ", length*sizeof(Type), " bytes, to dev_begin: ", dev_begin, " the last position is ", dev_begin + length );
        cuda_no_error_so_far();
        cuda_assert( cudaMemcpy( reinterpret_cast<void*>( dev_begin ), reinterpret_cast<const void*>( hst_begin ), length * sizeof( Type ), cudaMemcpyHostToDevice ) );
    }

    template<typename Type>
    void host_to_device( const Type* hst_begin, const Type* hst_end, Type* dev_begin )
    {
        std::size_t length = std::distance( hst_begin, hst_end );
        host_to_device_n( hst_begin, length, dev_begin );
    }

    template<typename Type>
    void device_to_host( const Type* dev, Type* hst )
    {
        cuda_assert( cudaMemcpy( reinterpret_cast<void*>( hst ), reinterpret_cast<const void*>( dev ), sizeof( Type ), cudaMemcpyDeviceToHost ) );
    }

    template<typename Type>
    void device_to_host_n( const Type* dev_begin, std::size_t length, Type* hst_begin )
    {
        cuda_assert( cudaMemcpy( reinterpret_cast<void*>( hst_begin ), reinterpret_cast<const void*>( dev_begin ), length * sizeof( Type ), cudaMemcpyDeviceToHost ) );
    }

    template<typename Type>
    void device_to_host( const Type* dev_begin, const Type* dev_end, Type* hst_begin )
    {
        std::size_t length = std::distance( dev_begin, dev_end );
        device_to_host_n( dev_begin, length, hst_begin );
    }

}//namespace ceras

extern "C"
{
    CUresult cudaMalloc ( void** devPtr, size_t size );
    CUresult cudaFree ( void* devPtr );
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

            //enlarge...
            //new_size += 1024 + (new_size >> 1);
            new_size += 1048576 + (new_size >> 1);

            if ( size_ > 0 )
                deallocate( data_ );

            data_ = allocate<T>( length );
            size_ = new_size;

            debug_print( "cuda_memory_cache reserves ", size_, " bytes, from ", static_cast<float*>(data_),  " to ", static_cast<float*>(data_)+(size_/sizeof(T)) );
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
        CUBLAS_OP_HERMITAN=2, /* synonym if CUBLAS_OP_C */
        CUBLAS_OP_CONJG=3     /* conjugate, placeholder - not supported in the current release */
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

    // cublas_handle& ch = singleton<cublas_handle>::instance();
    //
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

    void caffe_gpu_gemm_float(cublasHandle_t handle, const cublasOperation_t TransA, const cublasOperation_t TransB, const int M, const int N, const int K, const float alpha, const float* A, const float* B, const float beta, float* C)
    {
        int lda = (TransA == CUBLAS_OP_N) ? K : M;
        int ldb = (TransB == CUBLAS_OP_N) ? N : K;
        cublasOperation_t cuTransA = (TransA == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t cuTransB = (TransB == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublas_assert(cublasSgemm_v2(handle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
    }

    void caffe_gpu_gemm_double(cublasHandle_t handle, const cublasOperation_t TransA, const cublasOperation_t TransB, const int M, const int N, const int K, const double alpha, const double* A, const double* B, const double beta, double* C)
    {
        int lda = (TransA == CUBLAS_OP_N) ? K : M;
        int ldb = (TransB == CUBLAS_OP_N) ? N : K;
        cublasOperation_t cuTransA = (TransA == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t cuTransB = (TransB == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublas_assert(cublasDgemm_v2(handle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
    }

    // C <= A * B
    // where A or A' is [m x n], B or B' is [n x k] and C is [m x k]
    template< typename T > requires std::floating_point<T>
    void cuda_gemm( T const* A, bool a_transposed, T const* B, bool b_transposed, std::size_t m, std::size_t n, std::size_t k, T* C )
    {
        debug_print( "calling cuda_gemm with A=", A, ", B=", B, ", C=", C, ", m=", m, ", n=", n, ", k=", k, ", b_transposed=", b_transposed, ", a_transposed=", a_transposed );
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
        host_to_device_n( C, m*k, c );

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
            // C = A B, where C[m][k], A[m][n], B[n][k]
            // in fortran view, this is C^T[k][m] = B^T[k][n] A^T[n][m]
            // ldc of C and C^T is k, ldb of B and B^T is k, lda of A and A^T is n
            // row of C[m][k]is m, col of C[m][k] is k, the common dimmension is n
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

}

/*
template <>
void caffe_gpu_gemm<float>(const cublasOperation_t TransA, const cublasOperation_t TransB, const int M, const int N, const int K, const float alpha, const float* A, const float* B, const float beta, float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CUBLAS_OP_N) ? K : M;
  int ldb = (TransB == CUBLAS_OP_N) ? N : K;
  cublasOperation_t cuTransA = (TransA == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm_v2(Caffe::cublas_handle(), cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const cublasOperation_t TransA,
    const cublasOperation_t TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CUBLAS_OP_N) ? K : M;
  int ldb = (TransB == CUBLAS_OP_N) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm_v2(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}
*/

#endif//HLUDCUYMXCGREGWXHAVFNPYCNJLENNKSIOVPKKRXDUBYCAEMYBOKVKWODFGLJOUYYLQEDWSTB

