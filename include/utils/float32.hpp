#ifndef HJIVPPAHNJFYJXFXSWJYMPFQYJAXQGRRTRALWWRXVQNWRXTSCRQJULWOOQVESCPONOKLFGEAT
#define HJIVPPAHNJFYJXFXSWJYMPFQYJAXQGRRTRALWWRXVQNWRXTSCRQJULWOOQVESCPONOKLFGEAT

#include "../includes.hpp"
#include "./string.hpp"

namespace ceras
{
    namespace details
    {
        constexpr float string2float32( const char* s ) noexcept
        {
            auto const& isdigit = []( char c ) { return ( c >= '0' && c <= '9' ); };
            auto const& isspace = []( char c ) { return c == ' '; };

            float a = 0.0;
            int e = 0;
            int c;
            bool positive = true;

            while (*s && isspace(*s))
                ++s;

            if (*s == '+')
                ++s;

            if (*s == '-')
            {
                positive = false;
                ++s;
            }

            while ( ( c = *s++ ) != '\0' && isdigit( c ) )
            {
                a = a * 10.0 + ( c - '0' );
            }

            if ( c == '.' )
            {
                while ( ( c = *s++ ) != '\0' && isdigit( c ) )
                {
                    a = a * 10.0 + ( c - '0' );
                    e = e - 1;
                }
            }

            if ( c == 'e' || c == 'E' )
            {
                int sign = 1;
                int i = 0;
                c = *s++;

                if ( c == '+' )
                    c = *s++;
                else if ( c == '-' )
                {
                    c = *s++;
                    sign = -1;
                }

                while ( isdigit( c ) )
                {
                    i = i * 10 + ( c - '0' );
                    c = *s++;
                }

                e += i * sign;
            }

            while ( e > 0 )
            {
                a *= 10.0;
                e--;
            }

            while ( e < 0 )
            {
                a *= 0.1;
                e++;
            }

            return positive ? a : -a;
        }

    }//namespace details

//
// no compiler support template like 'a<1.03> a_;', though it is valid in c++20
// this is an temporary replacement
//
//template< ceras::float32 F="1.0" >
//struct vx
//{
//    static constexpr float val = F;
//
//    void operator()() const noexcept
//    {
//        std::cout << val << std::endl;
//    }
//};
//


template< unsigned N >
struct float32
{
    char buf_[N + 1] {};

    constexpr float32( char const* s ) noexcept
    {
        std::copy( s, s+N, buf_ );
    }

    constexpr operator float() const noexcept
    {
        return details::string2float32( buf_ );
    }
};

template<unsigned N> float32( char const ( & )[N] ) -> float32 < N - 1 >;





}//namespace ceras
#endif//HJIVPPAHNJFYJXFXSWJYMPFQYJAXQGRRTRALWWRXVQNWRXTSCRQJULWOOQVESCPONOKLFGEAT

