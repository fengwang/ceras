#include "utils/imageio.hpp"
ceras::tensor<unsigned char> resize_other(ceras::tensor<unsigned char> const& image) {
    auto shape=image.shape();return ceras::imageio::imresize(image,shape);
}
