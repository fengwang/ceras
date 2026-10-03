#pragma once
#include "../includes.hpp"
#include "../config.hpp"
#include "../utils/checked_size.hpp"
#ifdef CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif
namespace ceras {
#ifdef CUDA
inline void check_cuda(cudaError_t status) {
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
inline void check_cublas(cublasStatus_t status) {
    if(status!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(static_cast<int>(status)));
}
#endif
inline void cuda_no_error_so_far() {
#ifdef CUDA
    check_cuda(cudaGetLastError());
#endif
}
template<class T> T* allocate(std::size_t n) {
#ifdef CUDA
    void* p=nullptr;check_cuda(cudaMalloc(&p,checked_multiply(n,sizeof(T))));return static_cast<T*>(p);
#else
    throw std::runtime_error("CUDA backend disabled");
#endif
}
template<class T> void deallocate(T* p) {
#ifdef CUDA
    check_cuda(cudaFree(p));
#else
    if(p) throw std::runtime_error("CUDA backend disabled");
#endif
}
template<class T> T* allocate_host(std::size_t n) {
#ifdef CUDA
    void* p=nullptr;check_cuda(cudaMallocHost(&p,checked_multiply(n,sizeof(T))));return static_cast<T*>(p);
#else
    throw std::runtime_error("CUDA backend disabled");
#endif
}
template<class T> void deallocate_host(T* p) {
#ifdef CUDA
    check_cuda(cudaFreeHost(p));
#else
    if(p) throw std::runtime_error("CUDA backend disabled");
#endif
}
template<class T> void host_to_device_n(T const* source,std::size_t n,T* dest) {
#ifdef CUDA
    check_cuda(cudaMemcpy(dest,source,checked_multiply(n,sizeof(T)),cudaMemcpyHostToDevice));
#else
    throw std::runtime_error("CUDA backend disabled");
#endif
}
template<class T> void device_to_host_n(T const* source,std::size_t n,T* dest) {
#ifdef CUDA
    check_cuda(cudaMemcpy(dest,source,checked_multiply(n,sizeof(T)),cudaMemcpyDeviceToHost));
#else
    throw std::runtime_error("CUDA backend disabled");
#endif
}
template<class T> void host_to_device(T const* first,T const* last,T* dest) {host_to_device_n(first,last-first,dest);}
template<class T> void device_to_host(T const* first,T const* last,T* dest) {device_to_host_n(first,last-first,dest);}
#ifdef CUDA
struct cuda_backend_context {
    int device;
    cudaStream_t stream=nullptr;
    cublasHandle_t handle=nullptr;
    void* storage=nullptr;
    std::size_t capacity=0;
    std::mutex mutex;
    explicit cuda_backend_context(int d) : device(d) {
        int previous;check_cuda(cudaGetDevice(&previous));check_cuda(cudaSetDevice(device));
        try {
            check_cuda(cudaStreamCreate(&stream));check_cublas(cublasCreate(&handle));
            check_cublas(cublasSetStream(handle,stream));
        } catch(...) {
            if(handle)cublasDestroy(handle);if(stream)cudaStreamDestroy(stream);
            cudaSetDevice(previous);throw;
        }
        check_cuda(cudaSetDevice(previous));
    }
    cuda_backend_context(cuda_backend_context const&)=delete;
    cuda_backend_context& operator=(cuda_backend_context const&)=delete;
    ~cuda_backend_context() {
        int previous=0;cudaGetDevice(&previous);cudaSetDevice(device);
        if(stream)cudaStreamSynchronize(stream);
        if(storage)cudaFree(storage);if(handle)cublasDestroy(handle);if(stream)cudaStreamDestroy(stream);
        cudaSetDevice(previous);
    }
    void reserve(std::size_t bytes) {
        if(bytes<=capacity)return;
        void* replacement=nullptr;check_cuda(cudaMalloc(&replacement,bytes));
        if(storage)cudaFree(storage);storage=replacement;capacity=bytes;
    }
};
inline cuda_backend_context& default_cuda_context() {
    static thread_local std::unordered_map<int,std::unique_ptr<cuda_backend_context>> contexts;
    auto& context=contexts[visible_device];
    if(!context)context=std::make_unique<cuda_backend_context>(visible_device);
    return *context;
}
template<std::floating_point T>
void cuda_gemm(T const* A,bool ta,T const* B,bool tb,std::size_t m,std::size_t n,std::size_t k,T* C,cuda_backend_context& context) {
    static_assert(std::same_as<T,float> || std::same_as<T,double>);
    auto mi=checked_backend_dimension(m),ni=checked_backend_dimension(n),ki=checked_backend_dimension(k);
    auto an=checked_multiply(m,n),bn=checked_multiply(n,k),cn=checked_multiply(m,k);
    if(!m||!k)return;
    if(!C || (n && (!A||!B)))throw std::invalid_argument("null GEMM buffer");
    if(!n){std::fill_n(C,cn,T{});return;}
    auto total=checked_add(checked_add(an,bn),cn);
    std::lock_guard lock(context.mutex);
    int previous;check_cuda(cudaGetDevice(&previous));check_cuda(cudaSetDevice(context.device));
    struct restore_device {int previous; cudaStream_t stream; ~restore_device(){cudaStreamSynchronize(stream);cudaSetDevice(previous);}} restore{previous,context.stream};
    context.reserve(checked_multiply(total,sizeof(T)));
    auto a=static_cast<T*>(context.storage),b=a+an,c=b+bn;
    check_cuda(cudaMemcpyAsync(a,A,checked_multiply(an,sizeof(T)),cudaMemcpyHostToDevice,context.stream));
    check_cuda(cudaMemcpyAsync(b,B,checked_multiply(bn,sizeof(T)),cudaMemcpyHostToDevice,context.stream));
    T alpha=1,beta=0;
    if constexpr(std::same_as<T,float>)
        check_cublas(cublasSgemm(context.handle,tb?CUBLAS_OP_T:CUBLAS_OP_N,ta?CUBLAS_OP_T:CUBLAS_OP_N,ki,mi,ni,&alpha,b,tb?ni:ki,a,ta?mi:ni,&beta,c,ki));
    else
        check_cublas(cublasDgemm(context.handle,tb?CUBLAS_OP_T:CUBLAS_OP_N,ta?CUBLAS_OP_T:CUBLAS_OP_N,ki,mi,ni,&alpha,b,tb?ni:ki,a,ta?mi:ni,&beta,c,ki));
    check_cuda(cudaMemcpyAsync(C,c,checked_multiply(cn,sizeof(T)),cudaMemcpyDeviceToHost,context.stream));
    check_cuda(cudaStreamSynchronize(context.stream));
}
#endif
template<std::floating_point T>
void cuda_gemm(T const* A,bool ta,T const* B,bool tb,std::size_t m,std::size_t n,std::size_t k,T* C) {
#ifdef CUDA
    cuda_gemm(A,ta,B,tb,m,n,k,C,default_cuda_context());
#else
    throw std::runtime_error("CUDA backend disabled");
#endif
}
}
