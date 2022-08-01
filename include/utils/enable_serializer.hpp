#ifndef HYCBVEQFSDADSADKVPHMHKRXPNEVODNAVXQACROFCJMIKFNBOQCFEVAGDEKVCXPNBQQJPXVTB
#define HYCBVEQFSDADSADKVPHMHKRXPNEVODNAVXQACROFCJMIKFNBOQCFEVAGDEKVCXPNBQQJPXVTB

#include "../includes.hpp"
#include "./fmt.hpp"

namespace ceras
{

    struct default_unary_expression_serializer
    {

        template< typename Unary_Expression, typename Input_Expression >
        std::tuple<std::string, std::vector<std::string>> const operator()( Unary_Expression const& unary_expression, Input_Expression const& input_expression ) const noexcept
        {
            auto const& [ie_name, ie_code] = serialize( input_expression );
            //auto const& [ie_name, ie_code] = input_expression.serialize();
            std::string unary_expressionidentity = fmt::format( "unary_expression_{}_{}", unary_expression.name(), unary_expression.id() );
            std::vector<std::string> unary_expressioncode = ie_code;
            unary_expressioncode.emplace_back( fmt::format( "auto {} = {}({});", unary_expressionidentity, unary_expression.name(), ie_name ) );
            return std::make_tuple( unary_expressionidentity, unary_expressioncode );
        }

    }; // struct default_unary_expression_serializer

    struct default_binary_expression_serializer
    {
        template< typename Binary_Expression, typename Lhs_Expression, typename Rhs_Expression >
        std::tuple<std::string, std::vector<std::string>> const operator()( Binary_Expression const& binary_expression, Lhs_Expression const& lhs_input_expression, Rhs_Expression const& rhs_input_expression ) const noexcept
        {
            auto const& [lhs_ex_name, lhs_ex_code] = serialize( lhs_input_expression );
            //auto const& [lhs_ex_name, lhs_ex_code] = lhs_input_expression.serialize();
            auto const& [rhs_ex_name, rhs_ex_code] = serialize( rhs_input_expression );
            //auto const& [rhs_ex_name, rhs_ex_code] = rhs_input_expression.serialize();

            std::string binary_expression_identity = fmt::format( "binary_expression_{}_{}", binary_expression.name(), binary_expression.id() );

            std::vector<std::string> binary_expression_code = lhs_ex_code;
            std::copy( rhs_ex_code.begin(), rhs_ex_code.end(), std::back_inserter( binary_expression_code ) );

            binary_expression_code.emplace_back( fmt::format( "auto {} = {}( {}, {} );", binary_expression_identity, binary_expression.name(), lhs_ex_name, rhs_ex_name ) );

            return std::make_tuple( binary_expression_identity, binary_expression_code );
        }

    }; // struct default_binary_expression_serializer

    template< typename Unary_Expression >
    struct enable_unary_serializer
    {
        std::tuple< std::string, std::vector<std::string> > const serialize() const
        {
            auto const& zen = static_cast<Unary_Expression const&>( *this );
            return zen.serializer()( zen, zen.op() );
        }
    };

    template< typename Binary_Expression >
    struct enable_binary_serializer
    {
        std::tuple< std::string, std::vector<std::string> > const serialize() const
        {
            auto const& zen = static_cast<Binary_Expression const&>( *this );
            return zen.serializer()( zen, zen.lhs_op(), zen.rhs_op() );
        }
    };

}//namespace ceras

#endif//HYCBVEQFSDADSADKVPHMHKRXPNEVODNAVXQACROFCJMIKFNBOQCFEVAGDEKVCXPNBQQJPXVTB

