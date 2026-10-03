#pragma once
#include "../includes.hpp"
#include "../config.hpp"
#include "../utils/checked_size.hpp"
#ifdef CBLAS
#include <cblas.h>
#endif
namespace ceras {
template<std::floating_point T>
void cblas_gemm(T const* a,bool ta,T const* b,bool tb,std::size_t m,std::size_t n,std::size_t k,T* c) {
#ifdef CBLAS
    static_assert(std::same_as<T,float> || std::same_as<T,double>);
    auto mi=checked_backend_dimension(m),ni=checked_backend_dimension(n),ki=checked_backend_dimension(k);
    checked_multiply(checked_multiply(m,n),sizeof(T));checked_multiply(checked_multiply(n,k),sizeof(T));
    auto count=checked_multiply(m,k);checked_multiply(count,sizeof(T));
    if(!m||!k)return;
    if(!c || (n && (!a||!b)))throw std::invalid_argument("null GEMM buffer");
    if(!n){std::fill_n(c,count,T{});return;}
    if constexpr(std::same_as<T,float>)
        cblas_sgemm(CblasRowMajor,ta?CblasTrans:CblasNoTrans,tb?CblasTrans:CblasNoTrans,mi,ki,ni,1,a,ta?mi:ni,b,tb?ni:ki,0,c,ki);
    else
        cblas_dgemm(CblasRowMajor,ta?CblasTrans:CblasNoTrans,tb?CblasTrans:CblasNoTrans,mi,ki,ni,1,a,ta?mi:ni,b,tb?ni:ki,0,c,ki);
#else
    throw std::runtime_error("CBLAS backend disabled");
#endif
}
}
