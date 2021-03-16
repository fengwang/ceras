#ifndef MDDTPHCVUNGLJHIAQADCAPLLAATQQEDNOFBWRHKMAFHAROKBVMNQRDHYOXRSPULHMAEIPTPOE
#define MDDTPHCVUNGLJHIAQADCAPLLAATQQEDNOFBWRHKMAFHAROKBVMNQRDHYOXRSPULHMAEIPTPOE

#include "./string.hpp"

namespace ceras
{

    namespace ceras_private
    {
        struct id
        {
            int value_;
            constexpr id( int value = 0 ) noexcept: value_{value} {}
        };
    };//namespace ceras_private

    // return id sequentially
    inline int generate_uid() noexcept
    {
        static ceras_private::id id_generator;
        int ans = id_generator.value_;
        ++id_generator.value_;
        return ans;
    }

    template< typename Base, string Name="Anonymous Class"  >
    struct enable_id
    {
        //char const * name_ = Name;
        std::string name_ = std::string{Name};
        int id_;
        enable_id() noexcept : id_ { generate_uid() } {}
    };

}//namespace ceras

#endif//MDDTPHCVUNGLJHIAQADCAPLLAATQQEDNOFBWRHKMAFHAROKBVMNQRDHYOXRSPULHMAEIPTPOE

