#ifndef HCEYMEBBNMPNQQYAGTJLCRUWJFCNCMTYWJBKNHDTAHQVJPNVQHIFYUHHEUXBQSXTMIVUGJLJC
#define HCEYMEBBNMPNQQYAGTJLCRUWJFCNCMTYWJBKNHDTAHQVJPNVQHIFYUHHEUXBQSXTMIVUGJLJC

#include "../includes.hpp"

namespace ceras
{

#if 0
    template< std::weakly_incrementable W >
    constexpr auto range( W val_begin, W val_end )
    {
        return std::ranges::iota_view( val_begin, val_end );
    }

    template< std::weakly_incrementable W >
    constexpr auto range( W val_end )
    {
        return range( W{0}, val_end );
    }
#else

    template <typename T>
    struct detailed_range
    {
        typedef T                                                 value_type;
        typedef detailed_range                                             self_type;
        typedef std::function<value_type( value_type )>           step_function_type;
        typedef std::function<bool( value_type, value_type )>     equality_check_function_type;

        value_type                                                first;
        value_type                                                last;
        step_function_type                                        step_function;
        equality_check_function_type                              equality_check_function;

        struct iterator;

        //[first, last)
        template<typename Step_Function, typename Equality_Check_Function>
        detailed_range( value_type const& first_, value_type const& last_, Step_Function const& sf_, Equality_Check_Function const& ecf_ ) noexcept : first( first_ ), last( last_ ), step_function( sf_ ), equality_check_function( ecf_ ) {}

        template<typename Step_Function>
        detailed_range( value_type const& first_, value_type const& last_, Step_Function const& sf_ ) noexcept : first( first_ ), last( last_ ), step_function( sf_ ),  equality_check_function( []( value_type const& lhs, value_type const& rhs ) { return lhs == rhs; } ) {}

        detailed_range( value_type const& first_, value_type const& last_ ) noexcept : first( first_ ), last( last_ ), step_function( []( value_type x ) { return ++x; } ), equality_check_function( []( value_type const& lhs, value_type const& rhs ) noexcept { return lhs == rhs; } ) {}

        detailed_range( value_type const& last_ ) noexcept : first( 0 ), last( last_ ), step_function( []( value_type x ) { return ++x; } ), equality_check_function( []( value_type const& lhs, value_type const& rhs ) noexcept { return lhs == rhs; } ) {}

        iterator begin() const noexcept
        {
            return iterator{ first, step_function, equality_check_function };
        }

        iterator end() const noexcept
        {
            return iterator{ last, step_function, equality_check_function };
        }

        iterator cbegin() const noexcept
        {
            return iterator{ first, step_function, equality_check_function };
        }

        iterator cend() const noexcept
        {
            return iterator{ last, step_function, equality_check_function };
        }

    };//struct detailed_range

    template <typename T >
    struct detailed_range<T>::iterator
    {
        typedef iterator                                            self_type;
        typedef T                                                   value_type;
        typedef void                                                pointer;
        typedef void                                                reference;
        typedef std::size_t                                         size_type;
        typedef std::ptrdiff_t                                      difference_type;
        typedef std::input_iterator_tag                             iterator_category;
        typedef std::function<value_type( value_type )>             step_function_type;
        typedef std::function<bool( value_type, value_type )>       equality_check_function_type;

        value_type                                                  value;
        step_function_type                                          step_function;
        equality_check_function_type                                equality_check_function;

        iterator( value_type const& value_, step_function_type const& step_function_, equality_check_function_type const& equality_check_function_ ) noexcept : value( value_ ), step_function( step_function_ ), equality_check_function( equality_check_function_ ) {}

        value_type operator *() const noexcept
        {
            return value;
        }

        self_type& operator ++()
        {
            value = step_function(value);
            return *this;
        }

        self_type const operator ++(int)
        {
            self_type ans{*this};
            ++(*this);
            return ans;
        }

        self_type& operator +=( unsigned long  n )
        {
            while ( n--  )
                ++(*this);
            return *this;
        }

        friend self_type const operator + ( self_type const& lhs, unsigned long rhs )
        {
            self_type ans{ lhs };
            ans += rhs;
            return ans;
        }

        friend self_type const operator + ( unsigned long lhs, self_type const& rhs )
        {
            return rhs + lhs;
        }

        friend bool operator == ( self_type const& lhs, self_type const& rhs )
        {
            return  lhs.equality_check_function( lhs.value, rhs.value );
        }

        friend bool operator != ( self_type const& lhs, self_type const& rhs )
        {
            return !( lhs == rhs );
        }
    };//struct iterator


    template< typename T, typename Step_Function, typename Equality_Check_Function >
    detailed_range<T> const range( T const& first_, T const& last_, Step_Function const& sf_, Equality_Check_Function const& ecf_ )
    {
        return detailed_range<T>{ first_, last_, sf_, ecf_ };
    }

    template< typename T, typename Step_Function >
    detailed_range<T> const range( T const& first_, T const& last_, Step_Function const& sf_ )
    {
        return detailed_range<T>{ first_, last_, sf_ };
    }

    template< typename T >
    detailed_range<T> const range( T const& first_, T const& last_ )
    {
        return detailed_range<T>{ first_, last_ };
    }

    template< typename T >
    detailed_range<T> const range( T const& last_ )
    {
        return detailed_range<T>{ T{0}, last_ };
    }

    // make_iterator //
    //for_each( make_iterator{}, make_iterator{}, containter.begin(), ..., function );

#endif


}//namespace ceras

#endif//HCEYMEBBNMPNQQYAGTJLCRUWJFCNCMTYWJBKNHDTAHQVJPNVQHIFYUHHEUXBQSXTMIVUGJLJC

