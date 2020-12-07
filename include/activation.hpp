#ifndef DJDWJBHNDAYTNOXLFOBDSGAQAAYPWMXJGEBYIRKEAKAQUUWVGDUGGDKSDXUKSPCYYNTWTDNII
#define DJDWJBHNDAYTNOXLFOBDSGAQAAYPWMXJGEBYIRKEAKAQUUWVGDUGGDKSDXUKSPCYYNTWTDNII

#include "./operation.hpp"
#include "./tensor.hpp"
#include "./utils/range.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/for_each.hpp"

namespace ceras
{
    // TODO:
    // for expression/activation with only single input/output, the corresponding tensor can be reused without allocating new memory
    //

    template <Operation Op>
    auto constexpr softmax( Op const& op ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor x = deep_copy( input );
                                        std::size_t const last_dim = *(x.shape().rbegin());
                                        std::size_t const rest_dim = x.size() / last_dim;
                                        for ( auto idx : range( rest_dim ) )
                                        {
                                            auto [begin, end] = std::make_tuple( x.begin()+idx*last_dim, x.begin()+(idx+1)*last_dim );
                                            typename Tsor::value_type const mx = *std::max_element( begin, end );
                                            for_each( begin, end, [mx]( auto & v ){ v = std::exp( v-mx ); } );
                                            typename Tsor::value_type const sum = std::accumulate( begin, end, typename Tsor::value_type{0} );
                                            for_each( begin, end, [sum]( auto & v ){ v /= sum; } );
                                        }
                                        return x;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "backprop: upcoming gradient for activation softmax contains NaN" );
                                        Tsor ans{ output.shape() };
                                        for_each( ans.begin(), ans.end(), output.begin(), grad.begin(), []( auto & a, auto o, auto g ){ a = g * o * ( typename Tsor::value_type{1} - o ); } );
                                        return ans;
                                    }
                )( op );
    }

    template <Operation Op>
    auto constexpr sigmoid( Op const& op ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        auto ans = input.deep_copy();
                                        ans.map( [](auto& x){ x = 1.0 / (1.0+std::exp(-x)); } );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        Tsor ans{ output.shape() };
                                        for_each( ans.begin(), ans.end(), output.begin(), grad.begin(), []( auto & a, auto o, auto g ){ a = g * o * ( typename Tsor::value_type{1} - o ); } );
                                        return ans;
                                    }
                )( op );
    }

    template <Operation Op>
    auto constexpr tanh( Op const& op ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor ans = input.deep_copy();
                                        ans.map( [](auto& x){ x = 2.0 / (1.0+std::exp(-2.0*x)) - 1.0; } );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        auto ans = output.deep_copy();
                                        ans.map( []( auto& x ){ x = typename Tsor::value_type{1} - x * x; } );
                                        ans *= grad;
                                        return ans;
                                    }
                )( op );
    }

    template <Operation Op>
    auto constexpr relu( Op const& op ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor ans{ input.shape() };
                                        for ( auto idx : range( ans.size() ) ) // 1-D view of tensors input and ans
                                            ans[idx] = std::max( input[idx], typename Tsor::value_type{0} );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor ans = grad; // shallow copy
                                        const typename Tsor::value_type zero{0};
                                        for ( auto idx : range( ans.size() ) ) // 1-D view of tensors input, grad and ans
                                            ans[idx] = ( input[idx] > zero ) ? grad[idx] : zero;
                                        return ans;
                                    }
                )( op );
    }

}//namespace ceras

#endif//DJDWJBHNDAYTNOXLFOBDSGAQAAYPWMXJGEBYIRKEAKAQUUWVGDUGGDKSDXUKSPCYYNTWTDNII

