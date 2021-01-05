#ifndef SHARED_ANY_CAST_HPP_INCLUDED_AFSDIJLKSDF984ULAFJDLKJADSF98H4JFOIDJOJDJFJ
#define SHARED_ANY_CAST_HPP_INCLUDED_AFSDIJLKSDF984ULAFJDLKJADSF98H4JFOIDJOJDJFJ

#include "../includes.hpp"

namespace ceras
{
    template< typename T >
    T& context_cast( std::shared_ptr<std::any> cache )
    {
        static_assert( std::is_default_constructible_v<T>, "Error: type is not default constructible." );
        std::any& instance = *cache;
        if ( !instance.has_value() )
            instance = T{};
        return std::any_cast<T&>( instance );
    }

}//namespace ceras

#endif//SHARED_ANY_CAST_HPP_INCLUDED_AFSDIJLKSDF984ULAFJDLKJADSF98H4JFOIDJOJDJFJ

