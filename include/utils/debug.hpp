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

    template< typename ...M >
    inline void debug_print( M const& ... messages )
    {
        if constexpr (debug_mode)
        {
            auto&& do_print = []( std::ostream& os, auto... args ){ ( os << ... << color::rize(args, "Green") ); };
            std::cout << color::rize( "DEBUG", "Yellow" ) << ": ";
            do_print( std::cout, messages... );
            std::cout << std::endl;
        }
    }

    template< typename ... M >
    inline void debug_log( M const& ... message )
    {
        debug_print( message ... );
    }

}//namespace ceras

#endif//KXQAHRAFMICHTNHKCPYMXLKXBSQEVAKPNFBCWOGOMHSEEODJELRFHJQTJKXETLYXVUNPJRHRK
