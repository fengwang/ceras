#ifndef HCEYMEBBNMPNQQYAGTJLCRUWJFCNCMTYWJBKNHDTAHQVJPNVQHIFYUHHEUXBQSXTMIVUGJLJC
#define HCEYMEBBNMPNQQYAGTJLCRUWJFCNCMTYWJBKNHDTAHQVJPNVQHIFYUHHEUXBQSXTMIVUGJLJC

#include "../includes.hpp"
#include "./better_assert.hpp"

namespace ceras
{
    namespace range_impl_space_private_spdohjaslfk4y8hafasfjh4ohuyaf
    {
        template< typename T >
        struct ranger_iterator
        {
            typedef T                                       value_type;
            typedef std::random_access_iterator_tag         iterator_category;
            typedef std::ptrdiff_t                          difference_type;
            typedef ranger_iterator                         self_type;

            constexpr ranger_iterator( value_type value_, value_type stride_ ) noexcept: value(value_), stride( stride_ ) {}

            value_type value;
            value_type stride;

            constexpr value_type operator *() { return value; }

            constexpr self_type& operator ++() { value += stride; return *this; }
            constexpr self_type  operator ++(int) { self_type ans{*this}; value += stride; return ans; }

            constexpr self_type& operator --() { value -= stride; return *this; }
            constexpr self_type  operator --(int) { self_type ans{*this}; value -= stride; return ans; }

            constexpr self_type& operator +=( difference_type n ) { value += n*stride; return *this; }
            constexpr self_type& operator -=( difference_type n ) { value -= n*stride; return *this; }
        };//struct ranger_iterator

        template< typename T >
        constexpr bool operator != ( ranger_iterator<T> const& lhs, ranger_iterator<T> const& rhs )
        {
            return (lhs.value != rhs.value) || (lhs.stride != rhs.stride);
        }

        template< typename T >
        constexpr bool operator == ( ranger_iterator<T> const& lhs, ranger_iterator<T> const& rhs )
        {
            return (lhs.value == rhs.value) && (lhs.stride == rhs.stride);
        }

        template< typename T >
        constexpr auto operator + ( ranger_iterator<T> const& lhs, unsigned long n )
        {
            ranger_iterator<T> ans{ lhs };
            ans += n;
            return ans;
        }

        template< typename T >
        constexpr auto operator + ( unsigned long n, ranger_iterator<T> const& rhs )
        {
            return rhs + n;
        }

        template< typename T >
        constexpr auto operator - ( ranger_iterator<T> const& lhs, unsigned long n )
        {
            ranger_iterator<T> ans{ lhs };
            ans -= n;
            return ans;
        }

        template< typename T >
        constexpr auto operator - ( ranger_iterator<T> const& lhs, ranger_iterator<T> const& rhs )
        {
            better_assert( lhs.stride == rhs.stride, "stride mismatch!" );
            return static_cast<unsigned long>((lhs.value - rhs.value) / lhs.stride);
        }

        template< typename T >
        struct ranger
        {
            typedef T               value_type;
            typedef unsigned long   size_type;

            value_type              start;
            value_type              stop;
            value_type              stride;
            size_type               steps;

            constexpr ranger( value_type start_, value_type stop_, value_type stride_ )  : start(start_), stride(stride_)
            {
                steps = static_cast<size_type>( std::ceil((stop_-start_)/stride_) );
                stop = steps * stride + start;
            }

            constexpr auto begin() const { return ranger_iterator<T>{ start, stride }; }

            constexpr auto end() const { return ranger_iterator<T>{ stop, stride }; }

            constexpr value_type operator[]( unsigned long index ) const
            {
                better_assert( index < steps, "Out of range!" );
                return start + stride * index;
            }

            constexpr size_type size() const { return steps; }

        };//struct ranger

    }//namespace range_impl_space_private_spdohjaslfk4y8hafasfjh4ohuyaf

    template< typename T >
    constexpr auto range( T val_start, T val_end, T val_stride = T{1} )
    {
        //static_assert(std::is_pod<T>::value, "argument(s) to the function range must be POD." );
        return range_impl_space_private_spdohjaslfk4y8hafasfjh4ohuyaf::ranger{ val_start, val_end, val_stride };
    }

    template< typename T >
    constexpr auto range( T val_end )
    {
        return range( T{0}, val_end, T{1} );
    }

}//namespace ceras

#endif//HCEYMEBBNMPNQQYAGTJLCRUWJFCNCMTYWJBKNHDTAHQVJPNVQHIFYUHHEUXBQSXTMIVUGJLJC

