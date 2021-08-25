#ifndef NRFLVKIAQLDTRLNHHBYUJJAMYCRCFKLQTDSKDQSALHQGURGGKBSIGGVWXBSKHQGPAUDLPUBBQ
#define NRFLVKIAQLDTRLNHHBYUJJAMYCRCFKLQTDSKDQSALHQGURGGKBSIGGVWXBSKHQGPAUDLPUBBQ

#include "./includes.hpp"
#include "./tensor.hpp"
#include "./place_holder.hpp"
#include "./variable.hpp"
#include "./utils/singleton.hpp"
#include "./utils/debug.hpp"
#include "./utils/lzw.hpp"
#include "./utils/fmt.hpp"

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
        std::map<int, variable_type> variables_;

        session() { }

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
            if ( variables_.find( v.id_ ) == variables_.end() )
            {
                variables_.insert( {v.id_, v} );
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
            // find a tmp file
            //char* tmp_file_path = std::tmpnam( nullptr );
            std::string const& tmp_file_path =  file_path + std::string{".tmp"};

            // save original to tmp file
            save_original( tmp_file_path );

            // compress tmp file to file_path
            {
                std::ifstream ifs{ tmp_file_path, std::ios_base::binary };
                std::ofstream ofs( file_path, std::ios_base::binary );
                lzw::compress( ifs, ofs );
            }

            // remove original
            //std::remove( tmp_file_path );
            std::remove( tmp_file_path.c_str() );
        }

        void restore( std::string const& file_path )
        {
            // find a tmp file
            //char* tmp_file_path = std::tmpnam( nullptr );
            std::string const& tmp_file_path =  file_path + std::string{".tmp"};

            // uncompress tmp file
            {
                std::ifstream ifs( file_path, std::ios_base::binary );
                std::ofstream ofs{ tmp_file_path, std::ios_base::binary };
                lzw::decompress( ifs, ofs );
            }

            // restore original from tmp file to file_path
            restore_original( tmp_file_path );

            // remove tmp file
            //std::remove( tmp_file_path );
            std::remove( tmp_file_path.c_str() );
        }

        void save_original( std::string const& file_path ) const
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

        void restore_original( std::string const& file_path )
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

    template< Tensor Tsor >
    ceras_private::session<Tsor>& get_default_session()
    {
        return singleton<ceras_private::session<Tsor>>::instance();
    }

}//namespace ceras

#endif//NRFLVKIAQLDTRLNHHBYUJJAMYCRCFKLQTDSKDQSALHQGURGGKBSIGGVWXBSKHQGPAUDLPUBBQ

