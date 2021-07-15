#ifndef SJTXPMDBEYHNPQSGKFIEKTKOYWOFMOEGAHNOHMJVHJIAWTBHCCFUKCHLJMJAFPRHRXEOTYEDC
#define SJTXPMDBEYHNPQSGKFIEKTKOYWOFMOEGAHNOHMJVHJIAWTBHCCFUKCHLJMJAFPRHRXEOTYEDC

#include "./operation.hpp"

namespace ceras
{

    template< Expression Real_Ex, Expression Imag_Ex >
    struct complex
    {
        Real_Ex real_;
        Imag_Ex imag_;
    };//struct complex


    template< typename T >
    struct is_complex : std::false_type {};

    template< Expression Real_Ex, Expression Imag_Ex >
    struct is_complex<complex<Real_Ex, Imag_Ex>> : std::true_type {};

    template< typename T >
    constexpr bool is_complex_v = is_complex<T>::value;

    ///
    /// @concept Complex
    /// @brief A type that represents a complex operator.
    ///
    template< typename T >
    concept Complex = is_complex_v<T>;


    ///
    /// @bref Returns the real part of the complex operator.
    /// @param c A complex operator.
    ///
    template< Expression Real_Ex, Expression Imag_Ex >
    Real_Ex real( complex<Real_Ex, Imag_Ex> const& c ) noexcept
    {
        return c.real_;
    }

    ///
    /// @bref Returns the imaginary part of the complex operator.
    /// @param c A complex operator.
    ///
    template< Expression Real_Ex, Expression Imag_Ex >
    Real_Ex imag( complex<Real_Ex, Imag_Ex> const& c ) noexcept
    {
        return c.imag_;
    }


    // +, -, * with complex and Expression and a lot of functions

    ///
    /// @brief Returns the magnitude of the complex operator.
    /// @param c Complex operator.
    ///
    /// @code{.cpp}
    /// auto r = variable{ ... };
    /// auto i = variable{ ... };
    /// auto c = complex{ r, i };
    /// auto a = abs( c );
    /// @endcode
    ///
    template< Complex C >
    auto abs( C const& c ) noexcept
    {
        return hypot( real(c), imag(c) );
    }




}//namespace ceras

#endif//SJTXPMDBEYHNPQSGKFIEKTKOYWOFMOEGAHNOHMJVHJIAWTBHCCFUKCHLJMJAFPRHRXEOTYEDC

