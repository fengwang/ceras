#ifndef DGIDKOSTMBYAWNFFKALGHEJNQNOVEFFCRCJMOVVQQSNSJQPJMMTFBIVANGSXAWWUKGMEDNOAM
#define DGIDKOSTMBYAWNFFKALGHEJNQNOVEFFCRCJMOVVQQSNSJQPJMMTFBIVANGSXAWWUKGMEDNOAM

#include "../includes.hpp"

namespace Keras
{
    struct Input
    {
        std::vector<std::optional<unsigned long>> shape_;

        template< typename ... Integer_Or_NuLL>
        Input( Integer_Or_NuLL ... args )
        {
            shape_.reserve(sizeof...(args));
            (shape_.push_back(std::optional<unsigned long>{std::forward<decltype(args)>(args)}), ...);
        }
    };//Input

}//namespace Keras

#endif//DGIDKOSTMBYAWNFFKALGHEJNQNOVEFFCRCJMOVVQQSNSJQPJMMTFBIVANGSXAWWUKGMEDNOAM

