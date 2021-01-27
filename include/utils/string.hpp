#ifndef TJPXJFWFETKBKXILPMCFOVJYVEKATKTJSUNCECXINVHGJWKSJNVFJOPJPTKHBTGUGOTENKAVD
#define TJPXJFWFETKBKXILPMCFOVJYVEKATKTJSUNCECXINVHGJWKSJNVFJOPJPTKHBTGUGOTENKAVD

#include "../includes.hpp"

namespace ceras
{

template<unsigned N>
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
};

template<unsigned N> string( char const ( & )[N] ) -> string < N - 1 >;

}//namespace ceras

#endif//TJPXJFWFETKBKXILPMCFOVJYVEKATKTJSUNCECXINVHGJWKSJNVFJOPJPTKHBTGUGOTENKAVD

