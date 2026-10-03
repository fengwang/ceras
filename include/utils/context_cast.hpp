#ifndef SHARED_ANY_CAST_HPP_INCLUDED_AFSDIJLKSDF984ULAFJDLKJADSF98H4JFOIDJOJDJFJ
#define SHARED_ANY_CAST_HPP_INCLUDED_AFSDIJLKSDF984ULAFJDLKJADSF98H4JFOIDJOJDJFJ

#include "../includes.hpp"
#include "./debug.hpp"

namespace ceras
{

    using scratch_store=std::map<std::weak_ptr<std::any>,std::any,std::owner_less<std::weak_ptr<std::any>>>;
    inline thread_local scratch_store* active_scratch=nullptr;

    template< typename T >
    T& context_cast( std::shared_ptr<std::any> cache, T const& initial=T{} )
    {
        static_assert( std::is_default_constructible_v<T>, "Error: type is not default constructible." );
        std::any& instance = active_scratch ? (*active_scratch)[cache] : *cache;
        if ( !instance.has_value() )
            instance = initial;
        return std::any_cast<T&>( instance );
    }

    template< typename T >
    T& context_extract( std::shared_ptr<std::any> cache )
    {
        static_assert( std::is_default_constructible_v<T>, "Error: type is not default constructible." );
        std::any& instance = active_scratch ? (*active_scratch)[cache] : *cache;
        better_assert( instance.has_value(), "Error: expecting cache has cached a value, but got an empty instance." );
        return std::any_cast<T&>( instance );
    }

}//namespace ceras

#endif//SHARED_ANY_CAST_HPP_INCLUDED_AFSDIJLKSDF984ULAFJDLKJADSF98H4JFOIDJOJDJFJ

