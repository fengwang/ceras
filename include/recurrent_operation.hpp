#ifndef PGHFBYNPEXGPKRRAGXCDOHPEMDJEQMLAYAEQJEKEIFFEMGSIHWSEUPSPHGAJPJFYOBKFEKGWP
#define PGHFBYNPEXGPKRRAGXCDOHPEMDJEQMLAYAEQJEKEIFFEMGSIHWSEUPSPHGAJPJFYOBKFEKGWP

#include "./operation.hpp"
#include "./activation.hpp"
#include "./variable.hpp"

namespace ceras
{
    //
    // This operator copies the contents of lhs_ex to rhs_va, returns lhs_ex;
    // This operator is useful for recurrent operations to restore the update value
    //
    template< Expression Lhs_Expression, Variable Rhs_Variable>
    auto constexpr copy( Lhs_Expression const& lhs_ex, Rhs_Variable const& rhs_va ) noexcept // assign lhs value to rhs value
    {
        return make_binary_operator( []<Tensor Tsor>( Tsor const& lhs_tensor/*src*/, Tsor const& rhs_tensor/*dst*/ ) noexcept
        //return make_binary_operator( []<Tensor Tsor>( Tsor const& lhs_tensor/*src*/, Tsor& rhs_tensor/*dst*/ ) noexcept
                                     {
                                        auto& rhs = const_cast<Tsor&>(rhs_tensor);
                                        rhs.resize( lhs_tensor.shape() );
                                        std::copy( lhs_tensor.begin(), lhs_tensor.end(), rhs.begin() );
                                        return lhs_tensor;
                                        /*
                                        better_assert(  lhs_tensor.shape() == rhs_tensor.shape(), "copy: expecting src and dest shape agree, but lhs shape hase ",
                                                        lhs_tensor.ndim(), " dims with leading dimension ", *lhs_tensor.shape().begin(), " and rhs shape has ",
                                                        rhs_tensor.ndim(), " dims with leading dimension ", *rhs_tensor.shape().begin() );
                                        auto rhs_ptr = rhs_tensor.vector_ -> data();
                                        std::copy( lhs_tensor.begin(), lhs_tensor.end(), rhs_ptr );
                                        return lhs_tensor;
                                        */
                                     },
                                     []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                                     {
                                        return make_tuple( grad, zeros_like( grad ) ); // only src back-propagate
                                     },
                                     "Copy"
                )( lhs_ex, rhs_va );
    }

    //
    // example usage:
    //
    //  auto x = place_holder<Tsor>{}; // 1D input, of shape (batch_size, 16)
    //  auto l = lstm( 16, 32 )( x ); // creating a lstm layer
    //
    inline auto lstm = []( unsigned long input_size, unsigned long unit_size )
    {
        #warning "lstm is not well implemented yet"
        return [=]<Expression Ex>( Ex const& ex ) noexcept
        {
            typedef typename Ex::tensor_type tensor_type;
            typedef variable<tensor_type> variable_type;
            typedef typename tensor_type::value_type value_type;

            tensor_type tsor_h{ {unit_size,} }; // zero
            auto ht = variable_type{ tsor_h, false, true }; // statefule variable, not trainable

            auto hx = concatenate(-1)( ht, ex ); // size should be ( -1, unit_size+input_size )

            // f_t = \sigma (W_f * [h_t, x_t] + b_f)
            auto Wf = variable_type{ glorot_uniform<value_type>( {unit_size+input_size, unit_size} ) };
            auto bf = variable_type{ zeros<value_type>( { unit_size} ) };
            auto ft = sigmoid( hx * Wf + bf );

            // i_t = \sigma (W_i * [h_t, x_t] + b_i)
            auto Wi = variable_type{ glorot_uniform<value_type>( {unit_size+input_size, unit_size} ) };
            auto bi = variable_type{ zeros<value_type>( { unit_size,} ) };
            auto it = sigmoid( hx * Wi + bi );

            // c_t = \sigma (W_c * [h_t, x_t] + b_c)
            auto Wc = variable_type{ glorot_uniform<value_type>( {unit_size+input_size, unit_size} ) };
            auto bc = variable_type{ zeros<value_type>( { unit_size,} ) };
            auto ct = tanh( hx * Wc + bc );

            // o_t = \sigma (W_o * [h_t, x_t] + b_o)
            auto Wo = variable_type{ glorot_uniform<value_type>( {unit_size+input_size, unit_size} ) };
            auto bo = variable_type{ zeros<value_type>( {unit_size,} ) };
            auto ot = sigmoid( hx * Wo + bo );

            // C_t+1 = f_t * C_t + i_t * c_t
            tensor_type tsor_c{ {unit_size,} };
            auto C_t = variable_type{ tsor_c, false, true }; // stateful variable, not trainable
            auto C_t_1 = elementwise_product(ft, C_t) + elementwise_product(it, ct);
            auto CT = copy( C_t_1, C_t ); // update C_t for next iteration

            // h_t+1 = o_t * tahn(C_t+1)
            auto h_t_1 = elementwise_product( ot, tanh(CT) );
            auto HT = copy( h_t_1, ht ); // update h_t for next iteration

            return HT;
        };
    };

}//namespace ceras

#endif//PGHFBYNPEXGPKRRAGXCDOHPEMDJEQMLAYAEQJEKEIFFEMGSIHWSEUPSPHGAJPJFYOBKFEKGWP

