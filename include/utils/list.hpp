#ifndef VXKJIMQUPPWUWPBSGYAYNTPABARRUPRUUMSYIQNWMOFBSJKQACJPKCUADXPIDAHRDHPVVRIPY
#define VXKJIMQUPPWUWPBSGYAYNTPABARRUPRUUMSYIQNWMOFBSJKQACJPKCUADXPIDAHRDHPVVRIPY

namespace ceras
{

    template< typename ... Args >
    constexpr auto inline make_list( Args ... args ) noexcept
    {
        return [=](auto&& func) noexcept
        {
            return std::forward<decltype(func)>(func)( args... );
        };
    }

    template< typename ... Args >
    struct list
    {
        decltype( make_list( std::declval<Args>()... ) ) list_;

        template< typename ... Ts >
        constexpr list( Ts&&... args ) noexcept : list_{ make_list( std::forward<Ts>(args)... ) } {}
    };

    template< typename ... Args >
    constexpr auto car( list< Args ... > the_list )
    {
        return the_list( []( auto head, ... ) { return head; } );
    }

    template< typename Arg >
    constexpr auto car( list< Arg > the_list )
    {
        return the_list( []( auto head ) { return head; } );
    }

    constexpr auto car( list<> )
    {
        return make_list();
    }

    template< typename ... Args >
    constexpr auto cdr( list<Args...> the_list )
    {
        return the_list( []( auto, auto ... rests ){ return make_list( rests... ); } );
    }

    template< typename Arg >
    constexpr auto cdr( list<Arg> )
    {
        return make_list();
    }

    constexpr auto cdr( list<> )
    {
        return make_list();
    }

}//namespace ceras

#endif//VXKJIMQUPPWUWPBSGYAYNTPABARRUPRUUMSYIQNWMOFBSJKQACJPKCUADXPIDAHRDHPVVRIPY
