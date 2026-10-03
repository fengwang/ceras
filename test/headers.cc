#include "utils/cached_allocator.hpp"
#include "utils/buffered_allocator.hpp"
#include "utils/vector.hpp"
#include "utils/checked_size.hpp"
#include "tensor.hpp"
int main(){ceras::tensor<float> value({2},1.0f);return value.size()==2?0:1;}
