#ifndef PGHFBYNPEXGPKRRAGXCDOHPEMDJEQMLAYAEQJEKEIFFEMGSIHWSEUPSPHGAJPJFYOBKFEKGWP
#define PGHFBYNPEXGPKRRAGXCDOHPEMDJEQMLAYAEQJEKEIFFEMGSIHWSEUPSPHGAJPJFYOBKFEKGWP

#include "./operation.hpp"
#include "./activation.hpp"

namespace ceras
{

    template< Expression Lhs_Expression, Variable Rhs_Variable>
    auto constexpr copy( Lhs_Expression const& lhs_ex, Rhs_Variable const& rhs_va ) noexcept // assign lhs value to rhs value
    {
        return make_binary_operator( []<Tensor Tsor>( Tsor const& lhs_tensor/*src*/, Tsor const& rhs_tensor/*dst*/ ) noexcept
                                     {
                                        better_assert( lhs_tensor.shape() == rhs_tensor.shape(), "copy: expecting src and dest shape agree." );
                                        auto rhs_ptr = rhs_tensor.vector_ -> data();
                                        std::copy( lhs_tensor.begin(), lhs_tensor.end(), rhs_ptr );
                                        return lhs_tensor;
                                     },
                                     []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                                     {
                                        return make_tuple( grad, zeros_like( grad ) ); // only src back-propagate
                                     },
                                     "Copy"
                )( lhs_ex, rhs_va );
    }




}//namespace ceras

#endif//PGHFBYNPEXGPKRRAGXCDOHPEMDJEQMLAYAEQJEKEIFFEMGSIHWSEUPSPHGAJPJFYOBKFEKGWP

