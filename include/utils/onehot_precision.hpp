#ifndef ONEHOT_PRECISION_HPP_INCLUDED_SDLKJASLKJASLAKSDJLASDKJDFALKJSALSAKJSALKJSDLKJFLKJF
#define ONEHOT_PRECISION_HPP_INCLUDED_SDLKJASLKJASLAKSDJLASDKJDFALKJSALSAKJSALKJSDLKJFLKJF
#include "../tensor.hpp"
#include "./better_assert.hpp"

namespace ceras
{

    template< typename Ts > requires Tensor<Ts>
    double onehot_precision( Ts const& ground_truth, Ts const& prediction )
    {
        better_assert( ground_truth.shape() == prediction.shape() );
        auto const& gt = argmax( ground_truth, -1 );
        auto const& pd = argmax( prediction, -1 );
        auto const& relation = (gt == pd);
        auto const& rets = as_type<double>( relation );
        return mean( rets );
    }

}

#endif

