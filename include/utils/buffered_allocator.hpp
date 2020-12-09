#ifndef BUFFERED_ALLOCATOR_HPP_INCLUDED_DPSIOJASLKJ3489UASLIKJASOI8UJ3498UAFDSJA
#define BUFFERED_ALLOCATOR_HPP_INCLUDED_DPSIOJASLKJ3489UASLIKJASOI8UJ3498UAFDSJA

#include "../includes.hpp"

namespace ceras
{

    template< typename T, std::size_t BYTES > requires (not std::same_as<T, void>)
    struct buffered_allocator
    {
        typedef T value_type;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        constexpr buffered_allocator() noexcept {}

        constexpr buffered_allocator( const buffered_allocator& ) noexcept {}

        template< class U >
        constexpr buffered_allocator( const buffered_allocator<U, BYTES>& ) noexcept {}

        constexpr ~buffered_allocator() {}

        [[nodiscard]] constexpr T* allocate( std::size_t n )
        {
            const std::size_t bytes = sizeof(T) * n;

            if ( bytes <= BYTES )
                return reinterpret_cast<T*>( cache_.data() );

            std::allocator<T> a;
            return a.allocate( bytes );
        }

        constexpr void deallocate( T* p, std::size_t n )
        {
            const std::size_t bytes = sizeof(T) * n;

            if ( bytes <= BYTES )
                return;

            std::allocator<T> a;
            a.deallocate( p, n );
        }

        std::array<std::byte, BYTES> cache_;

        //althought this has been removed in std::allocator in C++20, but some STL implementation's allocator_trait relies on this class
        template< class U > struct rebind { typedef buffered_allocator<U, BYTES> other; };
    };

    template< class T1, class T2, std::size_t N >
    constexpr bool operator==( const buffered_allocator<T1, N>& lhs, const buffered_allocator<T2, N>& rhs ) noexcept
    {
        return lhs.cache_ == rhs.cache_;
    }

}//ceras

#endif//BUFFERED_ALLOCATOR_HPP_INCLUDED_DPSIOJASLKJ3489UASLIKJASOI8UJ3498UAFDSJA

