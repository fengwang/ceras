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
            constexpr id( int value = 0 ): value_{value} {}
        };
    };//namespace ceras_private

    // return id sequentially
    inline int generate_uid()
    {
        static std::atomic<int> next{0};
        return next.fetch_add(1, std::memory_order_relaxed);
    }

    template< typename Base, string Name="Anonymous Class"  >
    struct enable_id
    {
        //char const * name_ = Name;
        std::string name_ = std::string{Name};
        int id_;
        enable_id() : id_ { generate_uid() } {}

        int id() const
        {
            return id_;
        }

        std::string name() const
        {
            return name_;
        }
    };

}//namespace ceras

#endif//MDDTPHCVUNGLJHIAQADCAPLLAATQQEDNOFBWRHKMAFHAROKBVMNQRDHYOXRSPULHMAEIPTPOE

