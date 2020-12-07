#ifndef JWSTAWESJPDEXKGCYCJESXWYQEHAOVFVGMYWKYBLTCGWJLBSQIJCHASMPGUCMEQWUAWBBUBFK
#define JWSTAWESJPDEXKGCYCJESXWYQEHAOVFVGMYWKYBLTCGWJLBSQIJCHASMPGUCMEQWUAWBBUBFK

#include "../includes.hpp"
#include "./color.hpp"

namespace private_namespace
{   // for macro `better_assert`
    template< typename... Args >
    void print_assertion(std::ostream& out, Args&&... args)
    {
        if constexpr( debug_mode )
        {
            out.precision( 20 );
            (out << ... << args) << std::endl;
            abort();
        }
    }
}

#ifdef better_assert
#undef better_assert
#endif

//
// enhancing 'assert' macro, usage:
//
// int a;
// ...
// better_assert( a > 0 ); //same as 'assert'
// better_assert( a > 0, "a is expected larger than 0, but now a = " a ); //with more info dumped to std::cerr
//
//#define better_assert(EXPRESSION, ... ) ((EXPRESSION) ? (void)0 : private_namespace::print_assertion(std::cerr, "[Assertion Failure]: '", #EXPRESSION, "' in File: ", __FILE__, " in Line: ",  __LINE__ __VA_OPT__(,) __VA_ARGS__))
//#define better_assert(EXPRESSION, ... ) ((EXPRESSION) ? (void)0 : private_namespace::print_assertion(std::cerr, color::rize("Assertion Failure", "Red"), ": '", #EXPRESSION, "' in File: ", color::rize(__FILE__, "Cyan"), " in Line: ",  color::rize(__LINE__, "Blue"), " "  __VA_OPT__(,) __VA_ARGS__))
#define better_assert(EXPRESSION, ... ) ((EXPRESSION) ? (void)0 : private_namespace::print_assertion(std::cerr, color::rize("Assertion Failure", "Red", "Default", "Bold"), ": '", color::rize(#EXPRESSION, "Green"), "' in File: ", color::rize(__FILE__, "Yellow"), " in Line: ",  color::rize(__LINE__, "Blue"), " "  __VA_OPT__(,) __VA_ARGS__))

#endif//JWSTAWESJPDEXKGCYCJESXWYQEHAOVFVGMYWKYBLTCGWJLBSQIJCHASMPGUCMEQWUAWBBUBFK

