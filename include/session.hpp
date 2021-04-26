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

    namespace ceras_private
    {

    template< Tensor Tsor >
    struct session
    {
        typedef place_holder<Tsor> place_holder_type;
        typedef variable<Tsor> variable_type;
        typedef variable_state<Tsor> variable_state_type;

        std::vector<place_holder_type> place_holders_;
        //std::unordered_map<int, variable_type> variables_;
        std::map<int, variable_type> variables_;

        session()
        {
            //debug_log("!Creating a session!");
            //singleton<session<Tsor>*>::instance() = this;
        }

        session( session const& ) = delete;
        session( session&& ) = default;
        session& operator=( session const& ) = delete;
        session& operator=( session&& ) = default;

        void rebind( place_holder_type& p_holder, Tsor const& value )
        {
            p_holder.bind( value );
        }

        void bind( place_holder_type& p_holder, Tsor const& value )
        {
            p_holder.bind( value );
            place_holders_.emplace_back( p_holder );
        }

        void remember( variable_type const& v )
        {
            //debug_log( "trying to remember new varialble with id ", v.id_ );
            //debug_log( "session has ", variables_.size(), " variables remembered." );
            if ( variables_.find( v.id_ ) == variables_.end() )
            {
                variables_.insert( {v.id_, v} );
                //debug_log( "remembering new varialble with id ", v.id_ );
            }
        }

        template< typename Operation >
        auto run( Operation& op ) const
        {
            return op.forward();
        }

        // register variables associated to the op to this session
        // usually being called before restoring a session from a file
        template< typename Operation >
        void tap( Operation& op ) const
        {
            run( op );
        }

        void deserialize( std::string const& file_path )
        {
            restore( file_path );
        }

        void serialize( std::string const& file_path ) const
        {
            save( file_path );
        }

        void save( std::string const& file_path ) const
        {
            std::ofstream ofs{ file_path };
            better_assert( ofs.good(), "failed to open file ", file_path );

            // save id
            for ( auto const& [id, v] : variables_ )
            {
                ofs << id << " ";
            }
            ofs << "\n";

            // save tensors
            for ( auto const& [id, v] : variables_ )
            {
                write_tensor( ofs, v.data() );
            }

            ofs.close();
        }

        void restore( std::string const& file_path )
        {
            std::ifstream ifs{ file_path };
            better_assert( ifs.good(), "failed to open file ", file_path );

            // get list of ids from the 1st line
            std::vector<int> ids;
            {
                std::string str_ids;
                std::getline( ifs, str_ids );
                std::stringstream ss( str_ids );
                std::copy( std::istream_iterator<int>( ss ), std::istream_iterator<int>(), std::back_inserter( ids ) );
            }

            // restore each of the tensor, ignoring their gradients
            for ( auto id : ids )
            {
                auto itor = variables_.find( id );
                better_assert( itor != variables_.end(), "Error: unknown variable to load, the id is ", id );

                auto [_id, _var] = *itor;
                read_tensor( ifs, _var.data() );
            }

            ifs.close();
        }

        ~session()
        {
            for ( auto& p_holder : place_holders_ )
                p_holder.reset();

            place_holders_.clear();
            variables_.clear();

            singleton<session<Tsor>*>::instance() = nullptr;
        }
    }; // session

    } //namespace ceras_private
#if 0
    template< Tensor Tsor >
    std::reference_wrapper<ceras_private::session<Tsor>> get_default_session()
    {
        auto& sess = singleton<ceras_private::session<Tsor>>::instance();
        return std::ref(sess);
    }
#else
    template< Tensor Tsor >
    ceras_private::session<Tsor>& get_default_session()
    {
        return singleton<ceras_private::session<Tsor>>::instance();
    }
#endif

}//namespace ceras

#endif//NRFLVKIAQLDTRLNHHBYUJJAMYCRCFKLQTDSKDQSALHQGURGGKBSIGGVWXBSKHQGPAUDLPUBBQ

