#pragma once
#include <cstddef>
#include <limits>
#include <stdexcept>
namespace ceras {
inline constexpr std::size_t checked_add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max()-b) throw std::length_error("size overflow");
    return a+b;
}
inline constexpr std::size_t checked_multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max()/b)
        throw std::length_error("tensor extent/byte overflow");
    return a*b;
}
template<class Range>
constexpr std::size_t checked_elements(Range const& shape) {
    std::size_t n=1;
    for (auto d:shape) n=checked_multiply(n,d);
    return n;
}
inline int checked_backend_dimension(std::size_t n) {
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("backend dimension exceeds int");
    return static_cast<int>(n);
}
}
