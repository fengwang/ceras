#ifndef MDDTPHCVUNGLJHIAQADCAPLLAATQQEDNOFBWRHKMAFHAROKBVMNQRDHYOXRSPULHMAEIPTPOE
#define MDDTPHCVUNGLJHIAQADCAPLLAATQQEDNOFBWRHKMAFHAROKBVMNQRDHYOXRSPULHMAEIPTPOE

namespace ceras
{

    namespace ceras_private
    {
        struct id
        {
            int value_;
            id( int value = 0 ) noexcept: value_{value} {}
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

}//namespace ceras

#endif//MDDTPHCVUNGLJHIAQADCAPLLAATQQEDNOFBWRHKMAFHAROKBVMNQRDHYOXRSPULHMAEIPTPOE

