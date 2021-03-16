#ifndef NADEATFLLPQRFVEJOGGVTQHOSRXIJCPDESDIRCVRCLLGQBUXEKKODVSKVKBRCDOSMEFXEDNPB
#define NADEATFLLPQRFVEJOGGVTQHOSRXIJCPDESDIRCVRCLLGQBUXEKKODVSKVKBRCDOSMEFXEDNPB

#include "../loss.hpp"
#include "../operation.hpp"
#include "../utils/string.hpp"
#include "../utils/better_assert.hpp"

namespace Keras
{

    // example usage:
    //
    //     auto l = loss< "mae" >{};
    //
    template< ceras::string Name >
    struct loss
    {
        //static constexpr char const* name_ = Name;
        std::string name_ = Name;

        template< ceras::Expression Lhs_Expression, ceras::Expression Rhs_Expression >
        auto operator()( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) const noexcept
        {
            constexpr char const* static_name_ = Name;
            if constexpr ( ceras::string_equal( static_name_, "mae" ) || ceras::string_equal( static_name_, "mean_absolute_error") )
            {
                return ceras::mae( lhs_ex, rhs_ex );
            }
            else if constexpr ( ceras::string_equal( static_name_, "mse" ) || ceras::string_equal( static_name_, "mean_squared_error") )
            {
                return ceras::mse( lhs_ex, rhs_ex );
            }
            else if constexpr ( ceras::string_equal(static_name_, "binary_crossentropy") ||
                                ceras::string_equal(static_name_, "categorical_crossentropy") ||
                                ceras::string_equal(static_name_, "cross_entropy") ||
                                ceras::string_equal(static_name_, "crossentropy") )
            {
                ceras::debug_log( "Generating cross extropy loss from template name ", static_name_ );
                return ceras::cross_entropy_loss( lhs_ex, rhs_ex );
            }
            else
            {
                better_assert( false, "Not yet implemented this loss: ", static_name_ );
            }
        }
    };


}//namespace Keras

#endif//NADEATFLLPQRFVEJOGGVTQHOSRXIJCPDESDIRCVRCLLGQBUXEKKODVSKVKBRCDOSMEFXEDNPB

#if 0

TODO: losses from keras::loss


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 100. * K.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)

def sparse_categorical_crossentropy(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_pred, y_true)

def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1)


#endif

