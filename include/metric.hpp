#ifndef EIDQGGJMELOKMYSJNSKIUQTTOGNSMKBVITQSMTSVJDNWXPJBJKQBJDAIAIYMTPTIBLWPBCYLI
#define EIDQGGJMELOKMYSJNSKIUQTTOGNSMKBVITQSMTSVJDNWXPJBJKQBJDAIAIYMTPTIBLWPBCYLI

#include "./operation.hpp"
#include "./activation.hpp"
#include "./loss.hpp"

namespace ceras
{

    ///
    /// @brief
    ///
    template< Expression Lhs_Expression, Expression Rhs_Expression, std::floating_point FP >
    auto binary_accuracy( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex, FP threshold=0.5 ) noexcept
    {
        return mean( equal( lhs_ex, rhs_ex ) );
    }



}//namespace ceras

#endif//EIDQGGJMELOKMYSJNSKIUQTTOGNSMKBVITQSMTSVJDNWXPJBJKQBJDAIAIYMTPTIBLWPBCYLI

