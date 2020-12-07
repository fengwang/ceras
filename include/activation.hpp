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

    template <typename Op> requires Operation<Op>
    auto constexpr softmax( Op const& op ) noexcept
    {
        return make_unary_operator( []<typename T, typename A>( tensor<T,A> const& input ) noexcept
                                    {
                                        tensor<T,A> x = deep_copy( input );
                                        std::size_t const last_dim = *(x.shape().rbegin());
                                        std::size_t const rest_dim = x.size() / last_dim;
                                        for ( auto idx : range( rest_dim ) )
                                        {
                                            auto [begin, end] = std::make_tuple( x.begin()+idx*last_dim, x.begin()+(idx+1)*last_dim );
                                            T const mx = *std::max_element( begin, end );
                                            for_each( begin, end, [mx]( T& v ){ v = std::exp( v-mx ); } );
                                            T const sum = std::accumulate( begin, end, T{0} );
                                            for_each( begin, end, [sum]( T& v ){ v /= sum; } );
                                        }
                                        return x;
                                    },
                                    []<typename T, typename A>( tensor<T, A> const&, tensor<T, A> const& output, tensor<T, A> const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "backprop: upcoming gradient for activation softmax contains NaN" );
                                        tensor<T, A> ans{ output.shape() };
                                        for_each( ans.begin(), ans.end(), output.begin(), grad.begin(), []( T& a, T o, T g ){ a = g * o * ( T{1} - o ); } );
                                        return ans;
                                    }
                )( op );
    }

    template <typename Op> requires Operation<Op>
    auto constexpr sigmoid( Op const& op ) noexcept
    {
        return make_unary_operator( []<typename T, typename A>( tensor<T,A> const& input ) noexcept { auto ans = input.deep_copy(); ans.map( [](auto& x){ x = 1.0 / (1.0+std::exp(-x)); } ); return ans;},
                                    []<typename T, typename A>( tensor<T,A> const&, tensor<T,A> const& output, tensor<T,A> const& grad ) noexcept
                                    {
                                        tensor<T, A> ans{ output.shape() };
                                        for_each( ans.begin(), ans.end(), output.begin(), grad.begin(), []( T& a, T o, T g ){ a = g * o * ( T{1} - o ); } );
                                        return ans;
                                    }
                )( op );
    }

    template <typename Op> requires Operation<Op>
    auto constexpr tanh( Op const& op ) noexcept
    {
        return make_unary_operator( []<typename T, typename A>( tensor<T,A> const& input ) noexcept { tensor<T,A> ans = input.deep_copy(); ans.map( [](auto& x){ x = 2.0 / (1.0+std::exp(-2.0*x)) - 1.0; } ); return ans;},
                                    []<typename T, typename A>( tensor<T,A> const&, tensor<T,A> const& output, tensor<T,A> const& grad ) noexcept
                                    {
                                        auto ans = output.deep_copy();
                                        ans.map( []( auto& x ){ x = T{1} - x * x; } );
                                        ans *= grad;
                                        return ans;
                                    }
                )( op );
    }

    template <typename Op> requires Operation<Op>
    auto constexpr relu( Op const& op ) noexcept
    {
        return make_unary_operator( []<typename T, typename A>( tensor<T,A> const& input ) noexcept
                                    {
                                        tensor<T,A> ans{ input.shape() };
                                        for ( auto idx : range( ans.size() ) ) // 1-D view of tensors input and ans
                                            ans[idx] = std::max( input[idx], T{0} );
                                        return ans;
                                    },
                                    []<typename T, typename A>( tensor<T,A> const& input, tensor<T,A> const&, tensor<T,A> const& grad ) noexcept
                                    {
                                        tensor<T,A> ans = grad; // shallow copy
                                        for ( auto idx : range( ans.size() ) ) // 1-D view of tensors input, grad and ans
                                            ans[idx] = ( input[idx] > T{0} ) ? grad[idx] : T{0};
                                        return ans;
                                    }
                )( op );
    }

}//namespace ceras

#endif//DJDWJBHNDAYTNOXLFOBDSGAQAAYPWMXJGEBYIRKEAKAQUUWVGDUGGDKSDXUKSPCYYNTWTDNII

