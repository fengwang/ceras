#ifndef TJPXJFWFETKBKXILPMCFOVJYVEKATKTJSUNCECXINVHGJWKSJNVFJOPJPTKHBTGUGOTENKAVD
#define TJPXJFWFETKBKXILPMCFOVJYVEKATKTJSUNCECXINVHGJWKSJNVFJOPJPTKHBTGUGOTENKAVD

#include "../includes.hpp"

namespace ceras
{

    template< unsigned N >
    struct string
    {
        char buf[N + 1] {};

        constexpr string( char const* s ) noexcept
        {
            std::copy( s, s+N, buf );
        }

        constexpr operator char const* () const noexcept
        {
            return buf;
        }

        //constexpr operator std::string() const noexcept
        // FIXME: ‘std::__cxx11::basic_string<char>’ does not have ‘constexpr’ destructor  <-- gcc11
        operator std::string() const noexcept
        {
            return std::string{buf};
        }
    };

    template<unsigned N> string( char const ( & )[N] ) -> string < N - 1 >;

    // std::string's operator == is not constexpr
    inline constexpr bool string_equal( char const* b, char const* a ) noexcept
    {
        return (*a == *b) && (*a == '\0' || string_equal(a + 1, b + 1));
    }

}//namespace ceras

#endif//TJPXJFWFETKBKXILPMCFOVJYVEKATKTJSUNCECXINVHGJWKSJNVFJOPJPTKHBTGUGOTENKAVD

