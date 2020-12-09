#ifndef JPSBQEEADUPURARCCBVXOLXVMQHTNWCQWXUKKHCOTWFGOGXSODKEEYLSSTFGTVXNBROLKKEJM
#define JPSBQEEADUPURARCCBVXOLXVMQHTNWCQWXUKKHCOTWFGOGXSODKEEYLSSTFGTVXNBROLKKEJM

#include "./includes.hpp"
#include "./tensor.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/debug.hpp"

namespace ceras
{

    template< typename T, typename A=default_allocator<T> >
    struct place_holder
    {
        std::shared_ptr<tensor<T, A>> data_;

        place_holder() { }

        ~place_holder() { }

        tensor<T, A> const forward() const
        {
            better_assert( data_, "This place holder does not hold a data!" );
            return *data_;
        }

        void bind( tensor<T, A> const& data )
        {
            data_ = std::make_shared<tensor<T, A>>(data);
        }

        void reset()
        {
            data_.reset();
        }

        void backward( auto ) const noexcept { }
    };

    template< typename T >
    struct is_place_holder : std::false_type {};

    template< typename T, typename A >
    struct is_place_holder< place_holder< T, A> > : std::true_type {};

    template< class T >
    inline constexpr bool is_place_holder_v = is_place_holder<T>::value;

    template< typename T >
    concept Place_Holder = is_place_holder_v<T>;


}//namespace ceras

#endif//JPSBQEEADUPURARCCBVXOLXVMQHTNWCQWXUKKHCOTWFGOGXSODKEEYLSSTFGTVXNBROLKKEJM

