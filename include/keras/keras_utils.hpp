#ifndef HFPOPXPHNGPXJFBYBCHJKYKTXOQARCYOCLOLLOIADIUJUVQXLAGSIUUSCDYEMAXNDHDYJEHCP
#define HFPOPXPHNGPXJFBYBCHJKYKTXOQARCYOCLOLLOIADIUJUVQXLAGSIUUSCDYEMAXNDHDYJEHCP

#include "../includes.hpp"
#include "../utils/string.hpp"

namespace Keras
{

struct dummy { };

template<ceras::string T, typename Config>
struct string_config
{
    static constexpr char const* data_ = T;
};

template<ceras::string T="None">
struct activation : string_config< T, activation<T> > { };

template<ceras::string T="valid">
struct padding : string_config< T, padding<T> > { };

template<ceras::string T="glorot_uniform">
struct kernel_initializer : string_config< T, kernel_initializer<T> > { };

template<ceras::string T="zeros">
struct bias_initializer : string_config< T, bias_initializer<T> > { };

template<ceras::string T="None">
struct kernel_regularizer : string_config< T, kernel_regularizer<T> > { };

template<ceras::string T="None">
struct bias_regularizer : string_config< T, bias_regularizer<T> > { };

template<ceras::string T="None">
struct activity_regularizer : string_config< T, activity_regularizer<T> > { };

template<ceras::string T="None">
struct kernel_constraint : string_config< T, kernel_constraint<T> > { };

template<ceras::string T="None">
struct bias_constraint : string_config< T, bias_constraint<T> > { };

template< unsigned long N >
struct filters
{
    static constexpr unsigned long data_ =  N;
};

template< unsigned long M, unsigned long N=M >
struct kernel_size
{
    static constexpr std::tuple<unsigned long, unsigned long> data_ = {M, N};
};

template< unsigned long M, unsigned long N=M >
struct strides
{
    static constexpr std::tuple<unsigned long, unsigned long> data_ = {M, N};
};

template< unsigned long M, unsigned long N=M >
struct dilation_rate
{
    static constexpr std::tuple<unsigned long, unsigned long> data_ = {M, N};
};

template< bool b >
struct use_bias
{
    static constexpr bool data_ = b;
};


} //namespace Keras

#endif//

