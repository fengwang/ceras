#ifndef CACHED_ALLOCATOR_HPP_INCLUDED_FOPADIJASLDKJASLDFKJASDKLJFOIJFSAI84LKAFJF
#define CACHED_ALLOCATOR_HPP_INCLUDED_FOPADIJASLDKJASLDFKJASDKLJFOIJFSAI84LKAFJF

#include "../includes.hpp"
#include "../config.hpp"
#include "../backend/cuda.hpp"
#include "./singleton.hpp"

namespace ceras
{

    //---------------------------------------------------------
    //
    // cached_allocator is designed for Tensor only.
    //
    //---------------------------------------------------------


    namespace
    {
        struct memory
        {
            unsigned long size_;
            std::byte* address_;
            friend constexpr auto operator <=> ( memory const&, memory const& ) noexcept = default;
        }; // struct memory


        struct memory_hash
        {
            unsigned long operator()( memory const& mem ) const noexcept
            {
                return ( std::hash<unsigned long>()( mem.size_ ) << 1 ) ^ std::hash<std::byte*>()( mem.address_ );
            }
        }; // struct memory_hash

        struct memory_cache
        {
            std::unordered_set<memory, memory_hash> allocated_memory;
            std::unordered_multimap<unsigned long, std::byte*> reserved_memory;

            std::byte* allocate( unsigned long size )
            {
                // search reserved_memory, return if found one matches
                auto search = reserved_memory.find( size );
                if ( search != reserved_memory.end() )
                {
                    std::byte* ans = search->second;
                    allocated_memory.emplace( size, ans );
                    reserved_memory.erase( search );
                    return ans;
                }

                // create a fresh new memory
                std::allocator<std::byte> alloc;
                std::byte* ans = alloc.allocate( size );
                allocated_memory.emplace( size, ans );
                return ans;
            }

            void deallocate( std::byte* ptr, unsigned long size )
            {
                allocated_memory.erase( memory{size, ptr} );
                reserved_memory.emplace( size, ptr );
            }

            ~memory_cache()
            {
                std::allocator<std::byte> alloc;

                for ( auto& mem : allocated_memory )
                {
                    auto& [size, address] = mem;
                    alloc.deallocate( address, size );
                }

                for ( auto& [size, address] : reserved_memory )
                {
                    alloc.deallocate( address, size );
                }
            }

            void gc()
            {
                std::allocator<std::byte> alloc;
                for ( auto& [size, address] : reserved_memory )
                {
                    alloc.deallocate( address, size );
                }
            }

        }; // struct memory_cache

        memory_cache& get_memory_cache()
        {
            return singleton<memory_cache>::instance();
        }
    }//anonymous namespace

    //
    // Warning: only for tensor
    //
    template< typename T> requires (not std::same_as<T, void>)
    struct cached_allocator
    {
        typedef T value_type;
        typedef unsigned long size_type;
        typedef std::ptrdiff_t difference_type;

        constexpr cached_allocator() noexcept = default;
        constexpr cached_allocator( const cached_allocator<T>& ) noexcept = default;
        constexpr cached_allocator( cached_allocator<T>&& ) noexcept = default;

        template< class U >
        constexpr cached_allocator( const cached_allocator<U>& ) noexcept {}

        constexpr cached_allocator& operator = ( cached_allocator<T> const& ) noexcept = default;
        constexpr cached_allocator& operator = ( cached_allocator<T>&& ) noexcept = default;

        constexpr ~cached_allocator() {}

        [[nodiscard]] T* allocate( unsigned long const n )
        {
            std::byte* ans = get_memory_cache().allocate( n*sizeof(T) );
            std::memset( ans, 0, n*sizeof(T) );
            return reinterpret_cast<T*>( ans );
            //return reinterpret_cast<T*>( get_memory_cache().allocate( n*sizeof(T) ));
        }

        void deallocate( T* p, unsigned long const n )
        {
            get_memory_cache().deallocate( reinterpret_cast<std::byte*>(p), n * sizeof(T) );
        }

        // empty construct for performance reasons.
        template< typename U, typename ... Args >
        void construct( U*, Args&& ... )
        {
        }

        // empty construct_at for performance optimization reasons.
        template< typename U, typename ... Args>
        constexpr U* construct_at( U* ptr, Args&&... )
        {
            return ptr;
        }

        // empty destroy for performance reasons.
        template< typename U >
        void destroy( U* )
        {
        }

        //althought this has been removed in std::allocator from C++20 on, some STL's allocator_trait still relies on this embeded class
        template< class U > struct rebind { typedef cached_allocator<U> other; };
    };

    template< class T1, class T2 >
    constexpr bool operator==( const cached_allocator<T1>&, const cached_allocator<T2>& ) noexcept
    {
        return true;
    }

}//ceras

#endif//CACHED_ALLOCATOR_HPP_INCLUDED_FOPADIJASLDKJASLDFKJASDKLJFOIJFSAI84LKAFJF

