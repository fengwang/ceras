#ifndef KFCBQLIDARILSIWUSFKUFYYDWBMMJBYNVGGWYOBPAJTHJCKYLMQKHXSWIUTTLTBJQXQKXUJSS
#define KFCBQLIDARILSIWUSFKUFYYDWBMMJBYNVGGWYOBPAJTHJCKYLMQKHXSWIUTTLTBJQXQKXUJSS

#include "../includes.hpp"
#include "./fmt.hpp"
#include "./color.hpp"

namespace logging
{

    namespace logging_private
    {
        static int level = 0;

        template< typename ... Args >
        void impl_logging( std::string const& tag, std::string const& clor, std::string const& fm, int level,
                           std::string const& format_string, Args const& ... args )
        {
            if ( logging_private::level > level )
                return;

            auto const& message = fmt::format( format_string, args... );
            auto tm = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::cout << std::ctime(&tm);
            std::cout << fmt::format( "  [{}]: {}\n", color::rize(tag, clor, "Default", "Bold"), color::rize(message, clor, "Default", fm) );
        }

    } // namespace logging_private

    inline void set_level( int new_level )
    {
        logging_private::level = new_level;
    }

    template< typename ... Args >
    void log( std::string const& format_string, Args const& ... args )
    {
        logging_private::impl_logging( "LOG", "Blue", "Underlined", 0, format_string, args... );
    }

    template< typename ... Args >
    void debug( std::string const& format_string, Args const& ... args )
    {
        logging_private::impl_logging( "DEBUG", "Cyan", "Underlined", 10, format_string, args... );
    }

    template< typename ... Args >
    void info( std::string const& format_string, Args const& ... args )
    {
        logging_private::impl_logging( "INFO", "Green", "Underlined", 20, format_string, args... );
    }

    template< typename ... Args >
    void warning( std::string const& format_string, Args const& ... args )
    {
        logging_private::impl_logging( "WARNING", "Yellow", "Underlined", 30, format_string, args... );
    }

    template< typename ... Args >
    void error( std::string const& format_string, Args const& ... args )
    {
        logging_private::impl_logging( "ERROR", "Magenta", "Underlined", 40, format_string, args... );
    }

    template< typename ... Args >
    void critical( std::string const& format_string, Args const& ... args )
    {
        logging_private::impl_logging( "CRITICAL", "Red", "Underlined", 50, format_string, args... );
    }

}//namespace logging

#endif//KFCBQLIDARILSIWUSFKUFYYDWBMMJBYNVGGWYOBPAJTHJCKYLMQKHXSWIUTTLTBJQXQKXUJSS

