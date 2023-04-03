#ifndef QGIVVEUESCDFTUPFLDJSNBGBWHLCNAILBAKEBNCOGGOBHFQUQOTGENHFEHKQDGCDVWQSMJOQL
#define QGIVVEUESCDFTUPFLDJSNBGBWHLCNAILBAKEBNCOGGOBHFQUQOTGENHFEHKQDGCDVWQSMJOQL

#include "./constant.hpp"
#include "./session.hpp"
#include "./operation.hpp"
#include "./activation.hpp"
#include "./layer.hpp"

namespace ceras
{

    namespace
    {
        struct lstm_context
        {
            unsigned long units_;
        };



        template< Expression Ex, Expression Ey >
        auto copy_state( Ex const& ex, Ey const& ey ) noexcept
        {
            std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
            return make_binary_operator( [forward_cache]<Tensor Tsor>( Tsor const&, Tsor const& rhs_tensor ) noexcept
                                         {
                                            Tsor& ans = context_cast<Tsor>( forward_cache );
                                            ans.resize( rhs_tensor.shape() ); // note: when batch_size differes, the shape migh change
                                            std::copy( rhs_tensor.begin(), rhs_tensor.end(), ans.begin() );
                                            return ans;
                                         },
                                         []<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const& ) noexcept
                                         {
                                            auto z = zeros_like( lhs_input );
                                            return std::make_tuple( z, z );
                                         },
                                         "CopyState"
                    )( lhs_ex, rhs_ex );
        }

    }//anonymous namespace


    inline auto lstm( std::unsigned long units ) noexcept
    {
        std::shared_ptr<std::any> short_term_memory = std::make_shared<std::any>();
        std::shared_ptr<std::any> long_term_memory = std::make_shared<std::any>();

        return [=]<Expression Ex>( Ex const& ex ) noexcept
        {
            variable h_; // previous h
            varialbe c_; // previous c

            auto hx = concatenate( -1 )( h_, ex );
            auto f = sigmoid( Dense( units, units+units )( hx ) );
            auto i = sigmoid( Dense( units, units+units )( hx ) );
            auto ca = tanh( Dense( units, units+units )( hx ) );
            auto c = hadamard_product( f, c_ ) + hadamard_product( i, ca );

            auto o = sigmoid( Dense( units, units+units )( hx ) );
            auto h = hadamard_product( o, tanh( c ) );

            auto reserve_h = zeros_like( assign( h_, h ) );
            auto reserve_c = zeros_like( assign( c_, c ) );

            return o + reserve_h + reserve_c; // 'reserve_h' and 'reserve_c' are zeros. They are here just to prevent 'reserve_h' and 'reserve_c' from being optimized out.
        };
    };




}//namespace ceras

#endif//QGIVVEUESCDFTUPFLDJSNBGBWHLCNAILBAKEBNCOGGOBHFQUQOTGENHFEHKQDGCDVWQSMJOQL

