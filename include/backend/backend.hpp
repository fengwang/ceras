#ifndef EQJIONWMESOUCIKERIDHAQMPUFDHHUSGRVDSAXNXBOAHWIRPBOKRBMPWXNFUNYXGRYUULBLCG
#define EQJIONWMESOUCIKERIDHAQMPUFDHHUSGRVDSAXNXBOAHWIRPBOKRBMPWXNFUNYXGRYUULBLCG

#include "../includes.hpp"
#include "../config.hpp"

#include "./cuda.hpp"
#include "./cblas.hpp"

namespace ceras::backend
{

    ///
    /// @brief Allocate n object T.
    ///
    template< typename T >
    T* allocate( std::size_t n )
    {
        if constexpr( cuda_mode )
        {
            return ceras::allocate<T>( n );
        }
        else
        {
            return std::aligned_alloc( memory_alignment, n*sizeof(T) );
        }
    }

    ///
    /// @brief Deallocates the space previously allocated by ceras::backend::allocate()
    ///
    template< typename T >
    void deallocate( T* ptr )
    {
        if constexpr( cuda_mode )
        {
            ceras::deallocate<T>( size );
        }
        else
        {
            std::free( ptr );
        }
    }





}//namespace ceras::backend

#endif//EQJIONWMESOUCIKERIDHAQMPUFDHHUSGRVDSAXNXBOAHWIRPBOKRBMPWXNFUNYXGRYUULBLCG

