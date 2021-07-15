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
    /// @brief A type that represents a complex expression.
    ///
    template< typename T >
    concept Complex = is_complex_v<T>;


    ///
    /// @bref Returns the real part of the complex expression.
    /// @param c A complex expression.
    ///
    template< Expression Real_Ex, Expression Imag_Ex >
    Real_Ex real( complex<Real_Ex, Imag_Ex> const& c ) noexcept
    {
        return c.real_;
    }

    ///
    /// @bref Returns the imaginary part of the complex expression.
    /// @param c A complex expression.
    ///
    template< Expression Real_Ex, Expression Imag_Ex >
    Imag_Ex imag( complex<Real_Ex, Imag_Ex> const& c ) noexcept
    {
        return c.imag_;
    }


    // +, -, * with complex and Expression and a lot of functions

    ///
    /// @brief Returns the magnitude of the complex expression.
    /// @param c Complex expression.
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


    ///
    /// @brief Returns the complex expression.
    ///
    template< Complex C >
    auto operator + ( C const& c ) noexcept
    {
        return c;
    }

    ///
    /// @brief Negatives the complex expression.
    ///
    template< Complex C >
    auto operator - ( C const& c ) noexcept
    {
        return complex{ negative(real(c)), negative(imag(c)) };
    }


    ///
    /// @brief Sums up two complex expressions.
    ///
    template< Complex Cl, Complex Cr >
    auto operator + ( Cl const& cl, Cr const& cr ) noexcept
    {
        return complex{ real(cl)+real(cr), imag(cl)+imag(cr) };
    }


    ///
    /// @brief Subtracts one complex expression from the other one.
    ///
    template< Complex Cl, Complex Cr >
    auto operator - ( Cl const& cl, Cr const& cr ) noexcept
    {
        return cl + (-cr);
    }


    ///
    /// @brief Multiplies two complex expressions.
    /// Optimization here: (a+ib)*(c+id) = (ac-bd) + i(ad+bc) = (ac-bd) + i( (a+b)*(c+d)-ac-bd )
    ///
    /// @code{.cpp}
    /// auto c1 = complex{ ..., ... };
    /// auto c2 = complex{ ..., ... };
    /// auto c12 = c1 * c2;
    /// @endcode
    ///
    template< Complex Cl, Complex Cr >
    auto operator * ( Cl const& cl, Cr const& cr ) noexcept
    {
        auto const& a = real(cl);
        auto const& b = imag(cl);
        auto const& c = real(cr);
        auto const& d = imag(cr);
        auto const& ac = a * c;         // 1st multiplication
        auto const& bd = b * d;         // 2nd multiplication
        auto const& a_b = a + b;
        auto const& c_d = c + d;
        auto const& abcd = a_b * c_d;   // 3rd multiplication

        return complex{ ac-bd, abcd-ac-bd };
    }



    ///
    /// @brief Sums up a complex expression and an expression.
    ///
    template< Complex C, Expression E >
    auto operator + ( C const& c, E const& e ) noexcept
    {
        return complex{ real(c)+e, imag(c) };
    }

    ///
    /// @brief Sums up a complex expression and an expression.
    ///
    template< Complex C, Expression E >
    auto operator + ( E const& e, C const& c ) noexcept
    {
        return c + e;
    }


    ///
    /// @brief Subtracts an expression from a compression expression.
    ///
    template< Complex C, Expression E >
    auto operator - ( C const& c, E const& e ) noexcept
    {
        return complex{ real(c)-e, imag(c) };
    }

    ///
    /// @brief Subtractsa complex expression from an expression.
    ///
    template< Complex C, Expression E >
    auto operator - ( E const& e, C const& c ) noexcept
    {
        return c + e;
    }

    ///
    /// @brief Multiplies a complex expression with an expression.
    ///
    template< Complex C, Expression E >
    auto operator * ( C const& c, E const& e ) noexcept
    {
        return complex{ real(c)*e, imag(c)*e };
    }

    ///
    /// @brief Multiplies an expression with a compression expression.
    ///
    template< Complex C, Expression E >
    auto operator * ( E const& e, C const& c ) noexcept
    {
        return c * e;
    }



}//namespace ceras

#endif//SJTXPMDBEYHNPQSGKFIEKTKOYWOFMOEGAHNOHMJVHJIAWTBHCCFUKCHLJMJAFPRHRXEOTYEDC

