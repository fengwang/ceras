#ifndef JPSBQEEADUPURARCCBVXOLXVMQHTNWCQWXUKKHCOTWFGOGXSODKEEYLSSTFGTVXNBROLKKEJM
#define JPSBQEEADUPURARCCBVXOLXVMQHTNWCQWXUKKHCOTWFGOGXSODKEEYLSSTFGTVXNBROLKKEJM

#include "./includes.hpp"
#include "./tensor.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/debug.hpp"

namespace ceras
{

    template< Tensor Tsor >
    struct place_holder
    {
        std::shared_ptr<Tsor> data_;

        Tsor const forward() const
        {
            better_assert( data_, "This place holder does not hold a data!" );
            return *data_;
        }

        void bind( Tsor const& data )
        {
            data_ = std::make_shared<Tsor>(data);
        }

        void reset()
        {
            data_.reset();
        }

        // note: should not be written as 'void backward( ... ) const'
        // reason: cannot pass object of non-trivial type through variadic method
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

