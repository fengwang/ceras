#pragma once
#include "tensor.hpp"
namespace ceras::dataset {
namespace detail {
inline std::vector<std::uint8_t> read_idx(std::string const& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Cannot open IDX file: " + path);
    auto length=in.tellg();
    if (length<0 || static_cast<std::uintmax_t>(length)>tensor_io_limits.max_bytes)
        throw std::length_error("IDX file exceeds input limit");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
        throw std::runtime_error("Truncated IDX file: " + path);
    return bytes;
}
inline std::uint32_t field(std::span<std::uint8_t const> b, std::size_t offset) {
    if (offset>b.size() || b.size()-offset<4) throw std::invalid_argument("Short IDX header");
    std::uint32_t v=0;
    for(std::size_t i=0;i<4;++i) v=(v<<8)|b[offset+i];
    return v;
}
inline auto load_pair(std::string const& images, std::string const& labels) {
    auto x=read_idx(images),y=read_idx(labels);
    if(field(x,0)!=2051 || field(y,0)!=2049 || field(x,8)!=28 || field(x,12)!=28)
        throw std::invalid_argument("Invalid IDX magic or dimensions");
    auto n=field(x,4);
    if(field(y,4)!=n || x.size()-16!=checked_multiply(n,784) || y.size()-8!=n)
        throw std::invalid_argument("IDX count/payload mismatch");
    for(std::size_t i=8;i<y.size();++i)
        if(y[i]>=10) throw std::invalid_argument("IDX label outside [0,9]");
    tensor<std::uint8_t> input({n,28,28});
    tensor<std::uint8_t> target({n,10});
    std::copy(x.begin()+16,x.end(),input.begin());
    for(std::size_t i=0;i<n;++i) target[i*10+y[i+8]]=1;
    return std::make_pair(input,target);
}
inline auto load(std::string const& path) {
    auto [x,y]=load_pair(path+"/train-images-idx3-ubyte",path+"/train-labels-idx1-ubyte");
    auto [tx,ty]=load_pair(path+"/t10k-images-idx3-ubyte",path+"/t10k-labels-idx1-ubyte");
    return std::make_tuple(x,y,tx,ty);
}
}
namespace mnist {
// Images: [samples,28,28]; labels: one-hot [samples,10].
inline auto load_data(std::string const& path="./dataset/mnist") { return detail::load(path); }
}
namespace fashion_mnist {
inline auto load_data(std::string const& path="./dataset/fashion_mnist") { return detail::load(path); }
}
}
