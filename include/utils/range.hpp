#ifndef HCEYMEBBNMPNQQYAGTJLCRUWJFCNCMTYWJBKNHDTAHQVJPNVQHIFYUHHEUXBQSXTMIVUGJLJC
#define HCEYMEBBNMPNQQYAGTJLCRUWJFCNCMTYWJBKNHDTAHQVJPNVQHIFYUHHEUXBQSXTMIVUGJLJC

#include "../includes.hpp"

namespace ceras
{

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

}//namespace ceras

#endif//HCEYMEBBNMPNQQYAGTJLCRUWJFCNCMTYWJBKNHDTAHQVJPNVQHIFYUHHEUXBQSXTMIVUGJLJC

