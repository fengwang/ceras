#ifndef EADXAMBNTYNWIACMWPOXGCABMGJROMMVPHMOMRUMQDTQRDYTNYCGHKMIKVLKYJRHIKDRIXMCB
#define EADXAMBNTYNWIACMWPOXGCABMGJROMMVPHMOMRUMQDTQRDYTNYCGHKMIKVLKYJRHIKDRIXMCB

#include "../includes.hpp"
#include "./better_assert.hpp"

namespace ceras
{

    template< typename T, typename Alloc >
    struct vector
    {
        typedef unsigned long size_type;
        typedef long difference_type;
        typedef Alloc allocator_type;
        typedef T value_type;
        typedef value_type* pointer;
        typedef value_type* iterator;
        typedef const value_type* const_iterator;
        typedef value_type& reference;
        typedef const value_type& const_reference;

        unsigned long size_;
        pointer data_;

        constexpr vector( vector && other ) noexcept
        {
            size_ = other.size_;
            data_ = other.data_;
            other.size_ = 0;
            other.data_ = nullptr;
        }

        constexpr vector& operator = ( vector && other ) noexcept
        {
            size_ = other.size_;
            data_ = other.data_;
            other.size_ = 0;
            other.data_ = nullptr;
            return *this;
        }

        constexpr explicit vector( unsigned long size = 0, value_type init = value_type{} ) : size_{ 0 }, data_{ nullptr }
        {
            resize( size );

            if (!empty())
                std::fill_n( data_, size_, init );
        }

        constexpr vector( vector const& other ) : size_{ 0 }, data_{ nullptr }
        {
            resize( other.size() );
            if (!empty())
                std::copy_n( other.data_, size_, data_ );
        }

        constexpr vector& operator = ( vector const& other )
        {
            resize( other.size() );
            if (!empty())
                std::copy_n( other.data(), size(), data() );
            return *this;
        }

        constexpr vector( std::initializer_list<T> init ) : size_{0}, data_{ nullptr }
        {
            resize( init.size() );
            if ( !empty() )
                std::copy( init.begin(), init.end(), data() );
        }

        constexpr ~vector()
        {
            clear();
        }

        constexpr void clear()
        {
            if ( !empty() )
            {
                allocator_type alloc;
                alloc.deallocate( data_, size_ );
            }
            data_ = nullptr;
            size_ = 0;
        }

        constexpr void resize( unsigned long size )
        {
            if ( size == (*this).size() )
                return;

            clear();

            if ( size != 0 )
            {
                size_ = size;
                allocator_type alloc;
                data_ = alloc.allocate( size_ );
            }
        }

        constexpr pointer data() noexcept
        {
            return data_;
        }

        constexpr const_iterator data() const noexcept
        {
            return data_;
        }

        constexpr unsigned long size() const noexcept
        {
            return size_;
        }

        constexpr bool empty() const noexcept
        {
            return size() == 0;
        }


    }; // struct vector

}//namespace ceras

#endif//EADXAMBNTYNWIACMWPOXGCABMGJROMMVPHMOMRUMQDTQRDYTNYCGHKMIKVLKYJRHIKDRIXMCB

