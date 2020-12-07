#ifndef TWKJVHPVYJLTDSREHUKPFEUMOISXDIECEEHXNIWVCPENRBMGSEQFOWIQRGCXSHXCHVOOAQRAC
#define TWKJVHPVYJLTDSREHUKPFEUMOISXDIECEEHXNIWVCPENRBMGSEQFOWIQRGCXSHXCHVOOAQRAC

#include "../includes.hpp"

namespace ceras
{

    template <typename T, typename U>
    T const lexical_cast( U const& from )
    {
        T var;

        std::stringstream ss;
        ss << from;
        ss >> var;

        return var;
    }

}//namespace ceras

#endif//TWKJVHPVYJLTDSREHUKPFEUMOISXDIECEEHXNIWVCPENRBMGSEQFOWIQRGCXSHXCHVOOAQRAC
