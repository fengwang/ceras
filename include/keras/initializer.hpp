#ifndef MQEBSEDSAGUXWASAOIVVIUTLNCCEPFJBCHCVUHYCBYVIFKAEBJFYHGALOWYVIUSWEVCOTHAKH
#define MQEBSEDSAGUXWASAOIVVIUTLNCCEPFJBCHCVUHYCBYVIFKAEBJFYHGALOWYVIUSWEVCOTHAKH

#include "../includes.hpp"
#include "../tensor.hpp"
#include "../variable.hpp"
#include "../utils/string.hpp"
#include "../utils/float32.hpp"
#include "../utils/better_assert.hpp"

namespace Keras
{

    template< ceras::string Name="random_normal", ceras::float32 Arg1 = "0.0", ceras::float32 Arg2 = "1.0", ceras::float32 Arg3 = "0.0", ceras::float32 Arg4 = "0.0" >
    struct initializer
    {
        static constexpr char const * name_ = Name;
        static constexpr float arg1_ = Arg1;
        static constexpr float arg2_ = Arg2;
        static constexpr float arg3_ = Arg3;
        static constexpr float arg4_ = Arg4;

        auto constexpr operator()( std::vector<unsigned long> const& shape, unsigned long M=100, unsigned long N=100 ) const noexcept // M for fan_in, N for fan_out in case of glorot's approach
        {
            if constexpr ( ceras::string_equal( name_, "random_normal" ) )
            {
                return ceras::variable{ ceras::randn<float>( shape, arg1_, arg2_ ) } ;
            }
            else if constexpr ( ceras::string_equal( name_, "random_uniform" ) )
            {
                return ceras::variable{ ceras::random<float>( shape, arg1_, arg2_ ) } ;
            }
            else if constexpr ( ceras::string_equal( name_, "truncated_normal" ) )
            {
                return ceras::variable{ ceras::truncated_normal<float>( shape, arg1_, arg2_, arg3_, arg4_ ) } ;
            }
            else if constexpr ( ceras::string_equal( name_, "zeros" ) )
            {
                return ceras::variable{ ceras::zeros<float>( shape ) } ;
            }
            else if constexpr ( ceras::string_equal( name_, "ones" ) )
            {
                return ceras::variable{ ceras::ones<float>( shape ) } ;
            }
            else if constexpr ( ceras::string_equal( name_, "glorot_normal" ) )
            {
                const float variance = 2.0f / ( static_cast<float>(M) + static_cast<float>(N) );
                return ceras::variable{ ceras::randn<float>( shape, 0.0f, variance ) } ;

            }
            else if constexpr ( ceras::string_equal( name_, "glorot_uniform" ) )
            {
                const float limit = 6.0f / ( static_cast<float>(M) + static_cast<float>(N) );
                return ceras::variable{ ceras::random<float>( shape, -limit, limit ) } ;
            }
            else if constexpr ( ceras::string_equal( name_, "identity" ) )
            {
                better_assert( false, "Not implemented. This one is only for 2D case." );
                //TODO
                return ceras::variable{ ceras::random<float>( shape, arg1_, arg2_ ) } ;
            }
            else if constexpr ( ceras::string_equal( name_, "orthogonal" ) )
            {
                better_assert( false, "Not implemented yet. This one is only for 2D case." );
                return ceras::variable{ ceras::random<float>( shape, arg1_, arg2_ ) } ;
            }
            else if constexpr ( ceras::string_equal( name_, "constant" ) )
            {
                return ceras::variable{ ceras::tensor<float>{ shape, arg1_ } };
                /*
                constexpr ceras::tensor<float> x = ceras::ones<float>( shape );
                for ( auto& v : x ) v *= arg1_;
                return ceras::variable{ x };
                */
            }
            else // default case
            {
                better_assert( false, "Unknown initialzier: ", name_ );
                return ceras::variable{ ceras::zeros<float>( shape ) } ;
            }
        }
    };//initializer

}//namespace Keras
#endif//MQEBSEDSAGUXWASAOIVVIUTLNCCEPFJBCHCVUHYCBYVIFKAEBJFYHGALOWYVIUSWEVCOTHAKH

