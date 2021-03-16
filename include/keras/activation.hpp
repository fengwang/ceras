#ifndef WPTHOIGTDSEDKVNMKFAORASGLBHGGKDMUOODMINRMVCSKNXDRPMBYBQKUERKUWFKFIAQLYNHM
#define WPTHOIGTDSEDKVNMKFAORASGLBHGGKDMUOODMINRMVCSKNXDRPMBYBQKUERKUWFKFIAQLYNHM

#include "../includes.hpp"
#include "../activation.hpp"
#include "../utils/float32.hpp"
#include "../utils/string.hpp"

namespace Keras
{
    template< ceras::string Name >
    struct activation
    {
        std::string name_ = Name;

        template< ceras::Expression Ex>
        auto operator()( Ex const& ex ) const noexcept
        {
            constexpr char const * static_name = Name;
            if constexpr ( ceras::string_equal( static_name, "softmax" ) )
            {
                return ceras::softmax( ex );
            }
            else if constexpr( ceras::string_equal( static_name, "elu" ) )
            {
                return ceras::elu( 1.0 )( ex );
            }
            else if constexpr ( ceras::string_equal( static_name, "selu" ))
            {
                return ceras::selu( ex );
            }
            else if constexpr ( ceras::string_equal( static_name, "softplus" ))
            {
                return ceras::softmax( ex );
            }
            else if constexpr ( ceras::string_equal( static_name, "softsign" ))
            {
                return ceras::softsign( ex );
            }
            else if constexpr ( ceras::string_equal( static_name, "relu" ))
            {
                return ceras::relu( ex );
            }
            else if constexpr ( ceras::string_equal( static_name, "leaky_relu" ))
            {
                return ceras::leaky_relu(0.2)( ex );
            }
            else if constexpr ( ceras::string_equal( static_name, "tanh" ))
            {
                return ceras::tanh( ex );
            }
            else if constexpr ( ceras::string_equal( static_name, "sigmoid" ))
            {
                return ceras::sigmoid( ex );
            }
            else if constexpr ( ceras::string_equal( static_name, "hard_sigmoid" ))
            {
                return ceras::hard_sigmoid( ex );
            }
            else if constexpr ( ceras::string_equal( static_name, "exponential" ))
            {
                return ceras::exponential( ex );
            }
            else // fallback is 'linear'
            {
                return ex;
            }
        }
    };

}//namespace Keras

#endif//WPTHOIGTDSEDKVNMKFAORASGLBHGGKDMUOODMINRMVCSKNXDRPMBYBQKUERKUWFKFIAQLYNHM

