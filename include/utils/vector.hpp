#pragma once
#include <vector>
#include <memory>
namespace ceras {
// Tensor storage discards contents on a size change, unlike std::vector::resize.
template<class T, class Alloc>
class vector {
    std::vector<T, Alloc> storage_;
public:
    using value_type = T;
    using allocator_type = Alloc;
    vector() = default;
    explicit vector(std::size_t n, T value = T{}) : storage_(n, value) {}
    vector(std::initializer_list<T> values) : storage_(values) {}
    void resize(std::size_t n) {
        if (n != size()) {
            std::vector<T, Alloc> replacement(n, T{}, storage_.get_allocator());
            storage_.swap(replacement);
        }
    }
    void clear() noexcept { storage_.clear(); }
    T* data() noexcept { return storage_.data(); }
    T const* data() const noexcept { return storage_.data(); }
    std::size_t size() const noexcept { return storage_.size(); }
    bool empty() const noexcept { return storage_.empty(); }
};
}
