#ifndef KXQAHRAFMICHTNHKCPYMXLKXBSQEVAKPNFBCWOGOMHSEEODJELRFHJQTJKXETLYXVUNPJRHRK
#define KXQAHRAFMICHTNHKCPYMXLKXBSQEVAKPNFBCWOGOMHSEEODJELRFHJQTJKXETLYXVUNPJRHRK

#include "../includes.hpp"
#include "./color.hpp"
#include "./fmt.hpp"

namespace ceras
{

    namespace
    {
        inline std::string current_time() noexcept
        {
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            std::ostringstream oss;
            oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
            return oss.str();
        }
    }

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
    inline void debug_print( M const&  message, std::string const& fcolor=std::string{"Green"}, std::string const& background_color=std::string{"Default"}, std::string const& formatting=std::string{"Default"} )
    {
        if constexpr (debug_mode)
        {
            std::cout << color::rize( message, fcolor, background_color, formatting );
        }
    }


    template< typename ... MS >
    inline void debug_log( MS const& ... messages )
    {
        std::cout << color::rize( fmt::format("{} LOG ", current_time() ), "Green" ) << ": ";
        ( debug_print( messages ), ... );
    }

    // info
    template< typename ... MS >
    inline void debug_info( MS const& ... messages )
    {
        std::cout << color::rize( fmt::format("{} INFO ", current_time() ), "Green" ) << ": ";
        ( debug_print( messages ), ... );
    }

    // error
    template< typename ... MS >
    inline void debug_error( MS const& ... messages )
    {
        std::cout << color::rize( fmt::format("{} ERROR ", current_time() ), "Green" ) << ": ";
        ( debug_print( messages, std::string{"Red"}, std::string{"White"}, std::string{"Bold"} ), ... );
    }

    // warn
    template< typename ... MS >
    inline void debug_warn( MS const& ... messages )
    {
        std::cout << color::rize( fmt::format("{} WARN ", current_time() ), "Green" ) << ": ";
        ( debug_print( messages, std::string{"Yellow"}, std::string{"White"}, std::string{"Underlined"} ), ... );
    }

    // critical
    template< typename ... MS >
    inline void debug_critical( MS const& ... messages )
    {
        std::cout << color::rize( fmt::format("{} CRITICSL ", current_time() ), "Green" ) << ": ";
        ( debug_print( messages, std::string{"Blue"}, std::string{"White"}, std::string{"Reverse"} ), ... );
    }



}//namespace ceras

#endif//KXQAHRAFMICHTNHKCPYMXLKXBSQEVAKPNFBCWOGOMHSEEODJELRFHJQTJKXETLYXVUNPJRHRK
