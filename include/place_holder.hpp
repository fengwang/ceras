#ifndef JPSBQEEADUPURARCCBVXOLXVMQHTNWCQWXUKKHCOTWFGOGXSODKEEYLSSTFGTVXNBROLKKEJM
#define JPSBQEEADUPURARCCBVXOLXVMQHTNWCQWXUKKHCOTWFGOGXSODKEEYLSSTFGTVXNBROLKKEJM

#include "./includes.hpp"
#include "./tensor.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/debug.hpp"
#include "./utils/id.hpp"
#include "./utils/enable_shared.hpp"
#include "./utils/state.hpp"
#include "./utils/type2string.hpp"

namespace ceras
{

    template< Tensor Tsor >
    struct place_holder_state
    {
        Tsor data_;
        std::vector< unsigned long> shape_hint_;
    };

    template< Tensor Tsor >
    struct place_holder : enable_id< place_holder<Tsor>, "PlaceHolder" >, enable_shared_state<place_holder<Tsor>, place_holder_state<Tsor>>
    {
        typedef Tsor tensor_type;
        typedef typename tensor_type::value_type value_type;

        place_holder( place_holder const& other) = default;
        place_holder( place_holder && other) = default;
        place_holder& operator = ( place_holder const& other) = default;
        place_holder& operator = ( place_holder && other) = default;

        place_holder()
        {
            (*this).state_ = std::make_shared<place_holder_state<Tsor>>();
            (*((*this).state_)).shape_hint_ = std::vector< unsigned long >{ {-1UL,} };
        }

        place_holder( std::vector<unsigned long> const& shape_hint )
        {
            (*this).state_ = std::make_shared<place_holder_state<Tsor>>();
            (*((*this).state_)).shape_hint_ = shape_hint;
            if ( shape_hint[0] != -1UL )
            {
                auto & si = (*((*this).state_)).shape_hint_;
                si.insert( si.begin(), 1UL );
            }
        }

        void bind( Tsor data )
        {
            better_assert( (*this).state_, "Error with empty state." );
            (*((*this).state_)).data_ = data;
            (*((*this).state_)).shape_hint_ = data.shape();
        }

        Tsor const forward() const
        {
            better_assert( (*this).state_, "Error with empty state." );
            better_assert( !((*((*this).state_)).data_.empty()), "Error with empty tensor." );
            return (*((*this).state_)).data_;
        }

        void reset() noexcept
        {
            (*((*this).state_)).data_ = Tsor{};
            (*((*this).state_)).shape_hint_ = std::vector<unsigned long>{{-1UL}};
        }

        void backward( auto ) const noexcept { }

        void shape( std::vector< unsigned long> const& shape_hint ) noexcept
        {
            (*((*this).state_)).shape_hint_ = shape_hint;
        }

        std::vector<unsigned long> shape() const noexcept
        {
            if ( ! (*((*this).state_)).data_.empty() )
                return (*((*this).state_)).data_.shape();

            return (*((*this).state_)).shape_hint_;
        }

    };

    template< typename T >
    struct is_place_holder : std::false_type {};

    template< Tensor Tsor >
    struct is_place_holder< place_holder< Tsor > > : std::true_type {};

    template< class T >
    inline constexpr bool is_place_holder_v = is_place_holder<T>::value;

    template< typename T >
    concept Place_Holder = is_place_holder_v<T>;

    template< Place_Holder Ph >
    bool operator == ( Ph const& lhs, Ph const& rhs )
    {
        return lhs.id() == rhs.id();
    }

    template< Place_Holder Ph >
    bool operator != ( Ph const& lhs, Ph const& rhs )
    {
        return lhs.id() != rhs.id();
    }

    template< Place_Holder Ph >
    bool operator < ( Ph const& lhs, Ph const& rhs )
    {
        return lhs.id() < rhs.id();
    }

    template< Place_Holder Ph >
    bool operator > ( Ph const& lhs, Ph const& rhs )
    {
        return lhs.id() < rhs.id();
    }

    template< Place_Holder Ph >
    bool operator <= ( Ph const& lhs, Ph const& rhs )
    {
        return lhs.id() <= rhs.id();
    }

    template< Place_Holder Ph >
    bool operator >= ( Ph const& lhs, Ph const& rhs )
    {
        return lhs.id() >= rhs.id();
    }

    template< Place_Holder Ph >
    std::tuple<std::string, std::vector<std::string>> const serialize( Ph const& ph )
    {
        std::string const& ph_type = type2string<typename Ph::value_type>();
        std::string const& ph_name = fmt::format( "place_holder_{}", ph.id() );
        std::vector<std::string> ph_code;
        ph_code.emplace_back( fmt::format( "ceras::place_holder<ceras::tensor<{}>> {};\n", ph_type, ph_name ) );
        return std::forward_as_tuple( ph_name, ph_code );
    }

}//namespace ceras

#endif//JPSBQEEADUPURARCCBVXOLXVMQHTNWCQWXUKKHCOTWFGOGXSODKEEYLSSTFGTVXNBROLKKEJM

