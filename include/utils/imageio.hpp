#pragma once
#include "../tensor.hpp"
namespace ceras::imageio {
inline std::array<int,3> dimensions(std::vector<unsigned long> const& s) {
    if (s.size()!=2 && s.size()!=3) throw std::invalid_argument("image rank must be 2 or 3");
    int w=checked_backend_dimension(s[0]),h=checked_backend_dimension(s[1]);
    int c=s.size()==2?1:checked_backend_dimension(s[2]);
    if(w<=0 || h<=0 || c<1 || c>4) throw std::invalid_argument("invalid image dimensions/channels");
    return {w,h,c};
}
inline tensor<unsigned char> imresize(tensor<unsigned char> const& input, std::vector<unsigned long> const& shape) {
    auto [w,h,c]=dimensions(input.shape());
    auto [ow,oh,oc]=dimensions(shape);
    if(c!=oc || input.ndim()!=shape.size()) throw std::invalid_argument("image channel/rank mismatch");
    tensor<unsigned char> result(shape);
    if(!stbir_resize_uint8(input.data(),w,h,0,result.data(),ow,oh,0,c))
        throw std::runtime_error("image resize failed");
    return result;
}
inline tensor<unsigned char> imread(std::string const& path) {
    int w=0,h=0,c=0;
    std::unique_ptr<unsigned char,decltype(&stbi_image_free)> image(stbi_load(path.c_str(),&w,&h,&c,0),stbi_image_free);
    if(!image) throw std::runtime_error("Cannot read image: "+path);
    tensor<unsigned char> result({static_cast<unsigned long>(w),static_cast<unsigned long>(h),static_cast<unsigned long>(c)});
    std::copy_n(image.get(),result.size(),result.begin());
    return result;
}
template<Tensor Tsor>
bool direct_imwrite(std::string const& path, Tsor const& input) {
    auto [w,h,c]=dimensions(input.shape());
    auto ext=std::filesystem::path(path).extension().string();
    if(ext==".hdr") {
        tensor<float> pixels(input.shape());
        std::transform(input.begin(),input.end(),pixels.begin(),[](auto v){return static_cast<float>(v);});
        return stbi_write_hdr(path.c_str(),w,h,c,pixels.data())!=0;
    }
    if constexpr(!std::same_as<typename Tsor::value_type,unsigned char>) {
        throw std::invalid_argument("direct byte image writer requires unsigned char pixels");
    } else {
        if(ext==".bmp") return stbi_write_bmp(path.c_str(),w,h,c,input.data())!=0;
        if(ext==".tga") return stbi_write_tga(path.c_str(),w,h,c,input.data())!=0;
        if(ext==".jpg" || ext==".jpeg") return stbi_write_jpg(path.c_str(),w,h,c,input.data(),100)!=0;
        auto output=ext==".png"?path:path+".png";
        return stbi_write_png(output.c_str(),w,h,c,input.data(),0)!=0;
    }
}
template<Tensor Tsor>
bool imwrite(std::string const& path, Tsor const& input) {
    dimensions(input.shape());
    if(std::filesystem::path(path).extension()==".hdr") return direct_imwrite(path,input);
    if constexpr(std::same_as<typename Tsor::value_type,unsigned char>) return direct_imwrite(path,input);
    else {
        auto lo=amin(input),hi=amax(input);
        if(!std::isfinite(lo) || !std::isfinite(hi)) throw std::invalid_argument("nonfinite image pixels");
        tensor<unsigned char> bytes(input.shape());
        std::transform(input.begin(),input.end(),bytes.begin(),[=](auto v){
            if(!std::isfinite(v)) throw std::invalid_argument("nonfinite image pixels");
            return static_cast<unsigned char>(hi==lo?0:std::clamp(255.0*(v-lo)/(hi-lo),0.0,255.0));
        });
        return direct_imwrite(path,bytes);
    }
}
}
