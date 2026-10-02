#pragma once
#include <memory>
#include <cstddef>
namespace ceras {
// Compatibility name; standard allocation replaces the unsafe custom storage.
template<class T>
using cached_allocator = std::allocator<T>;
}
