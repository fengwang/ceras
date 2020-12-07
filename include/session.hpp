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

    template< typename T, typename A=std::allocator<T> >
    struct session
    {
        typedef tensor<T, A> tensor_type;
        typedef place_holder<T, A> place_holder_type;
        typedef variable<T, A> variable_type;

        std::vector<std::reference_wrapper<place_holder_type>> place_holders_;
        std::map<int, std::reference_wrapper<variable_type>> variables_;

        session()
        {
            singleton<session<T,A>*>::instance() = this;
        }

        session( session const& ) = delete;
        session( session&& ) = delete;
        session& operator=( session const& ) = delete;
        session& operator=( session&& ) = delete;

        void rebind( place_holder_type& p_holder, tensor_type const& value )
        {
            p_holder.bind( value );
        }

        void bind( place_holder_type& p_holder, tensor_type const& value )
        {
            p_holder.bind( value );
            place_holders_.emplace_back( std::ref( p_holder ) );
        }

        void remember( variable_type& v )
        {
            variables_.insert( {v.id_, std::ref(v)} );
        }

        template< typename Operation >
        auto run( Operation& op ) const
        {
            return op.forward();
        }

        ~session()
        {
            for ( auto& p_holder : place_holders_ )
                p_holder.get().reset();

            place_holders_.clear();
            variables_.clear();

            singleton<session<T,A>*>::instance() = nullptr;
        }
    };

    template< typename T, typename A >
    std::reference_wrapper<session<T,A>> get_default_session()
    {
        auto p_session = singleton<session<T,A>*>::instance();
        return std::ref(*p_session);
    }

}//namespace ceras

#endif//NRFLVKIAQLDTRLNHHBYUJJAMYCRCFKLQTDSKDQSALHQGURGGKBSIGGVWXBSKHQGPAUDLPUBBQ

