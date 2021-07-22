#ifndef KXQAHRAFMICHTNHKCPYMXLKXBSQEVAKPNFBCWOGOMHSEEODJELRFHJQTJKXETLYXVUNPJRHRK
#define KXQAHRAFMICHTNHKCPYMXLKXBSQEVAKPNFBCWOGOMHSEEODJELRFHJQTJKXETLYXVUNPJRHRK

#include "../includes.hpp"
#include "./color.hpp"

namespace ceras
{

    template< typename T >
    inline void debug_print( std::vector<T> const& message )
    {
        if constexpr (debug_mode)
        {
            std::cout << color::rize( "DEBUG(vector)", "Blue" ) << ": [ ";
            std::copy( message.begin(), message.end(), std::ostream_iterator<T>( std::cout, " " ) );
            std::cout << "]" << std::endl;
        }
    }

    template< typename M >
    inline void debug_print( M const&  message )
    {
        if constexpr (debug_mode)
        {
            std::cout << color::rize( "DEBUG", "Yellow" ) << ": ";
            std::cout << color::rize( message, "Green" );
            std::cout << std::endl;
        }
    }

    template< typename M >
    inline void debug_log( M const& m )
    {
        debug_print( m );
    }

    template< typename M, typename ... MS >
    inline void debug_log( M const& m, MS const& ... message )
    {
        debug_log( m );
        debug_log( message ... );
    }

}//namespace ceras

#endif//KXQAHRAFMICHTNHKCPYMXLKXBSQEVAKPNFBCWOGOMHSEEODJELRFHJQTJKXETLYXVUNPJRHRK
