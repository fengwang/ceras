#ifndef JPSBQEEADUPURARCCBVXOLXVMQHTNWCQWXUKKHCOTWFGOGXSODKEEYLSSTFGTVXNBROLKKEJM
#define JPSBQEEADUPURARCCBVXOLXVMQHTNWCQWXUKKHCOTWFGOGXSODKEEYLSSTFGTVXNBROLKKEJM

#include "./includes.hpp"
#include "./tensor.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/debug.hpp"
#include "./utils/id.hpp"
#include "./utils/enable_shared.hpp"
#include "./utils/state.hpp"

namespace ceras
{

    template< Tensor Tsor >
    struct place_holder_state
    {
        Tsor data_;
        std::vector< unsigned long> shape_hint_;
    };

    template< Tensor Tsor >
    struct place_holder :   enable_id< place_holder<Tsor>, "Place Holder" >,
                            enable_shared_state<place_holder<Tsor>, place_holder_state<Tsor>>
    {
        typedef Tsor tensor_type;

        place_holder( place_holder const& other) = default;
        place_holder( place_holder && other) = default;
        place_holder& operator = ( place_holder const& other) = default;
        place_holder& operator = ( place_holder && other) = default;

        place_holder()
        {
            (*this).state_ = std::make_shared<place_holder_state<Tsor>>();
        }

        place_holder( std::vector<unsigned long> const& shape_hint )
        {
            (*this).state_ = std::make_shared<place_holder_state<Tsor>>();
            (*((*this).state_)).shape_hint_ = shape_hint;
        }

        void bind( Tsor data )
        {
            // TODO: check data shape
            better_assert( (*this).state_, "Error with empty state." );
            (*((*this).state_)).data_ = data;
        }

        Tsor const forward() const
        {
            better_assert( (*this).state_, "Error with empty state." );
            better_assert( !((*((*this).state_)).data_.empty()), "Error with empty tensor." );
            return (*((*this).state_)).data_;
        }

        void reset() { }

        void backward( auto ) const noexcept { }
    };

    template< typename T >
    struct is_place_holder : std::false_type {};

    template< Tensor Tsor >
    struct is_place_holder< place_holder< Tsor > > : std::true_type {};

    template< class T >
    inline constexpr bool is_place_holder_v = is_place_holder<T>::value;

    template< typename T >
    concept Place_Holder = is_place_holder_v<T>;

}//namespace ceras

#endif//JPSBQEEADUPURARCCBVXOLXVMQHTNWCQWXUKKHCOTWFGOGXSODKEEYLSSTFGTVXNBROLKKEJM

