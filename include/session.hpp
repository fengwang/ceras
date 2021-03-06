#ifndef NRFLVKIAQLDTRLNHHBYUJJAMYCRCFKLQTDSKDQSALHQGURGGKBSIGGVWXBSKHQGPAUDLPUBBQ
#define NRFLVKIAQLDTRLNHHBYUJJAMYCRCFKLQTDSKDQSALHQGURGGKBSIGGVWXBSKHQGPAUDLPUBBQ

#include "./includes.hpp"
#include "./tensor.hpp"
#include "./place_holder.hpp"
#include "./variable.hpp"
#include "./utils/singleton.hpp"
#include "./utils/debug.hpp"

namespace ceras
{

    template< Tensor Tsor >
    struct session
    {
        typedef place_holder<Tsor> place_holder_type;
        typedef variable<Tsor> variable_type;

        //std::vector<std::reference_wrapper<place_holder_type>> place_holders_;
        //std::map<int, std::reference_wrapper<variable_type>> variables_;

        std::vector<place_holder_type> place_holders_;
        std::map<int, variable_type> variables_;


        session()
        {
            singleton<session<Tsor>*>::instance() = this;
        }

        session( session const& ) = delete;
        session( session&& ) = delete;
        session& operator=( session const& ) = delete;
        session& operator=( session&& ) = delete;

        void rebind( place_holder_type& p_holder, Tsor const& value )
        {
            debug_print( "session is rebinding place holder: ", p_holder.id_ );
            p_holder.bind( value );
        }

        void bind( place_holder_type& p_holder, Tsor const& value )
        {
            //debug_print( "session is binding place holder: ", p_holder.id_, " with tensor ", value.id_ );
            p_holder.bind( value );
            //place_holders_.emplace_back( std::ref( p_holder ) );
            place_holders_.emplace_back( p_holder );
        }

        void remember( variable_type& v )
        {
            //variables_.insert( {v.id_, std::ref(v)} );
            variables_.insert( {v.id_, v} );
        }

        template< typename Operation >
        auto run( Operation& op ) const
        {
            // debug here!
            {
                /*
                if constexpr( debug_mode )
                {
                    std::cout << "session has collected these place holders:\n";
                    //for ( auto& p : place_holders_ ) std::cout << p.get().id_ << "  ";
                    for ( auto& p : place_holders_ ) std::cout << p.id_ << "  ";
                    std::cout << std::endl;
                }
                */
            }
            return op.forward();
        }

        ~session()
        {
            for ( auto& p_holder : place_holders_ )
                //p_holder.get().reset();
                p_holder.reset();

            place_holders_.clear();
            variables_.clear();

            singleton<session<Tsor>*>::instance() = nullptr;
        }
    };

    template< Tensor Tsor >
    std::reference_wrapper<session<Tsor>> get_default_session()
    {
        auto p_session = singleton<session<Tsor>*>::instance();
        return std::ref(*p_session);
    }

}//namespace ceras

#endif//NRFLVKIAQLDTRLNHHBYUJJAMYCRCFKLQTDSKDQSALHQGURGGKBSIGGVWXBSKHQGPAUDLPUBBQ

