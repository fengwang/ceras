#ifndef EOOXHPVAAAEVBJXVOCEUKAMIYLTCRESXBYUTMOLJSHHGBNAWHOEPRBVFMPDNCGOPGWNUHKVQT
#define EOOXHPVAAAEVBJXVOCEUKAMIYLTCRESXBYUTMOLJSHHGBNAWHOEPRBVFMPDNCGOPGWNUHKVQT

#include "../includes.hpp"
#include "./debug.hpp"
#include "./fmt.hpp"

namespace ceras
{
    struct timer
    {
        typedef unsigned long clock_type;
        typedef float float_type;
        typedef double double_type;
        typedef long double long_double_type;
        std::clock_t t;

        template<typename T >
        timer( [[maybe_unused]]const T& val )
        {
            t = std::clock();
            //debug_log( fmt::format( "Timer begins with information \'{}\' at {}", val, t ) );
        }

        timer()
        {
            t = std::clock();
            //debug_log( fmt::format( "Timer begins at {}", t ) );
        }

        ~timer()
        {
            //debug_log( fmt::format( "Timer ends. Duration: {} seconds", static_cast<long double>(*this) ) );
        }

        operator clock_type () const
        {
            const std::clock_t _t = std::clock();
            const std::clock_t  d = _t - t;
            return d;
        }

        operator float_type () const
        {
            const std::clock_t _t = std::clock();
            const std::clock_t  d = _t - t;
            return  float_type(d) / CLOCKS_PER_SEC;
        }

        operator double_type () const
        {
            const std::clock_t _t = std::clock();
            const std::clock_t  d = _t - t;
            return  double_type(d) / CLOCKS_PER_SEC;
        }

        operator long_double_type () const
        {
            const std::clock_t _t = std::clock();
            const std::clock_t  d = _t - t;
            return  long_double_type(d) / CLOCKS_PER_SEC;
        }
    };

    template< typename Func >
    timer time_it( Func const& func )
    {
        timer t;
        func();
        return t;
    }

}//namespace ceras

#endif//EOOXHPVAAAEVBJXVOCEUKAMIYLTCRESXBYUTMOLJSHHGBNAWHOEPRBVFMPDNCGOPGWNUHKVQT

