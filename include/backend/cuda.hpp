#ifndef HLUDCUYMXCGREGWXHAVFNPYCNJLENNKSIOVPKKRXDUBYCAEMYBOKVKWODFGLJOUYYLQEDWSTB
#define HLUDCUYMXCGREGWXHAVFNPYCNJLENNKSIOVPKKRXDUBYCAEMYBOKVKWODFGLJOUYYLQEDWSTB

#include "../includes.hpp"
#include "../utils/singleton.hpp"


namespace ceras
{

    void set_device( int );

    template< typename T >
    T* allocate( unsigned long n )
    {
        void cuda_allocate( void** p, unsigned long n );
        T* ans;
        cuda_allocate( reinterpret_cast<void**>(&ans), n*sizeof(T) );
        return ans;
    }

    template< typename T >
    void deallocate( T* ptr )
    {
        void cuda_deallocate( void* p );
        cuda_deallocate( reinterpret_cast<void*>(ptr) );
    }

    template< typename T >
    void host_to_device_copy( T* host_begin, T* host_end, T* device_begin )
    {
        void cuda_memcopy_host_to_device( const void* src, unsigned long n, void* dst );
        unsigned long const n = sizeof(T)*(host_end-host_begin);
        cuda_memcopy_host_to_device( reinterpret_cast<const void*>(host_begin), n, reinterpret_cast<void*>(device_begin) );
    }

    template< typename T >
    void device_to_host_copy( T* device_begin, T* device_end, T* host_begin )
    {
        void cuda_memcopy_device_to_host( const void* src, unsigned long n, void* dst );
        unsigned long const n = sizeof(T)*(device_end-device_begin);
        cuda_memcopy_device_to_host( reinterpret_cast<const void*>(device_begin), n, reinterpret_cast<void*>(host_begin) );
    }

    struct cuda_memory_cache
    {
        int         device_id_;
        void*       data_;
        std::size_t size_in_bytes_;

        ~cuda_memory_cache()
        {
            set_device( device_id_ );
            deallocate( data_ );
            data_ = nullptr;
            size_in_bytes_ = 0;
        }
    };

    // TODO: should be implemented outside
    // 1. get instance of the cache
    // 2. check if current cache size larger than the already reserved cache or not
    // 3. if larger, then deallocate cache, and reallocate a larger one
    inline void cuda_memory_cache_reserve( std::size_t new_size_in_bytes )
    {
        cuda_memory_cache& cache = singleton<cuda_memory_cache>::instance();
        if ( cache.size_in_bytes_ >= new_size_in_bytes ) return;

        cache.data_ = allocate<void>( new_size_in_bytes );
        cache.size_in_bytes_ = new_size_in_bytes;
    }

}//namespace ceras

#endif//HLUDCUYMXCGREGWXHAVFNPYCNJLENNKSIOVPKKRXDUBYCAEMYBOKVKWODFGLJOUYYLQEDWSTB

