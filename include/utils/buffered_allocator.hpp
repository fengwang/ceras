#pragma once
#include <memory>
#include <cstddef>
namespace ceras {
// Compatibility name; standard allocation replaces the unsafe custom storage.
template<class T, std::size_t Bytes>
using buffered_allocator = std::allocator<T>;
}
