#ifndef FOR_EACH_HPP_INCLUDED_DSPOJSADLK111111111111111111111983UY4KAJSLFLKJFDIF
#define FOR_EACH_HPP_INCLUDED_DSPOJSADLK111111111111111111111983UY4KAJSLFLKJFDIF

#include "../includes.hpp"
#include "./range.hpp"

namespace ceras
{
namespace for_each_impl_private
{
    template < std::size_t Index, typename Type, typename... Types >
    struct extract_type_forward
    {
        typedef typename extract_type_forward < Index - 1, Types... >::result_type result_type;
    };

    template < typename Type, typename... Types >
    struct extract_type_forward< 1, Type, Types... >
    {
        typedef Type result_type;
    };

    template < typename Type, typename... Types >
    struct extract_type_forward< 0, Type, Types... >
    {
        struct index_parameter_for_extract_type_forwrod_should_not_be_0;
        typedef index_parameter_for_extract_type_forwrod_should_not_be_0 result_type;
    };

    template < std::size_t Index, typename... Types >
    struct extract_type_backward
    {
        typedef typename extract_type_forward < sizeof...( Types ) - Index + 1, Types... >::result_type result_type;
    };

    template < std::size_t Index, typename... Types >
    struct extract_type
    {
        typedef typename extract_type_forward< Index, Types... >::result_type result_type;
    };

    template < typename Function, typename InputIterator1, typename... InputIteratorn >
    constexpr Function _for_each_n( Function f, std::size_t n, InputIterator1 begin1, InputIteratorn... beginn )
    {
        for ( auto idx : range( n ) ) f( *(begin1+idx), *(beginn+idx)... );
        return f;
    }

    template < typename Function, typename InputIterator1, typename... InputIteratorn >
    constexpr Function _for_each( Function f, InputIterator1 begin1, InputIterator1 end1, InputIteratorn... beginn )
    {
        return _for_each_n( f, std::distance( begin1, end1 ), begin1, beginn... );
    }

    struct dummy { };

    template < typename... Types_N >
    struct for_each_impl_with_dummy
    {
        typedef typename extract_type_backward< 1, Types_N... >::result_type return_type;
        template < typename Predict, typename... Types >
        constexpr Predict impl( Predict p, dummy, Types... types ) const
        {
            return _for_each( p, types... );
        }
        template < typename S, typename... Types >
        constexpr return_type impl( S s, Types... types ) const
        {
            return impl( types..., s );
        }
    };
}//namespace for_each_impl_private

template < typename... Types >
constexpr typename for_each_impl_private::extract_type_backward< 1, Types... >::result_type
for_each( Types... types )
{
    static_assert( sizeof...( types ) > 2, "f::for_each requires at least 3 arguments" );
    return for_each_impl_private::for_each_impl_with_dummy< Types... >().impl( types..., for_each_impl_private::dummy() );
}

}//namespace ceras

#endif
