#ifndef HFPOPXPHNGPXJFBYBCHJKYKTXOQARCYOCLOLLOIADIUJUVQXLAGSIUUSCDYEMAXNDHDYJEHCP
#define HFPOPXPHNGPXJFBYBCHJKYKTXOQARCYOCLOLLOIADIUJUVQXLAGSIUUSCDYEMAXNDHDYJEHCP

#include "../includes.hpp"
#include "../utils/string.hpp"

namespace Keras
{

struct dummy { };

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

template< unsigned long ... Sizes >
struct shape
{
    static constexpr std::vector<unsigned long> data_ {{Sizes...}}; // or std::array??
};

template< unsigned long ... Sizes >
using target_shape = shape<Sizes...>;

template< typename Concrete_Type >
struct enable_shared_ptr: std::enable_shared_from_this<Concrete_Type>
{
    std::shared_ptr<Concrete_Type> get_shared_ptr()
    {
        auto& zen = static_cast<Concrete_Type&>(*this);
        return zen.shared_from_this();
    };
    std::shared_ptr<Concrete_Type> get_shared_ptr() const
    {
        auto const& zen = static_cast<Concrete_Type const&>(*this);
        return zen.shared_from_this();
    };
};

} //namespace Keras

#endif//

