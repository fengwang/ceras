#include "ceras.hpp"
#include "utils/imageio.hpp"
#include <atomic>
#include <stdexcept>

using ceras::tensor;
tensor<unsigned char> resize_other(tensor<unsigned char> const&);
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance=1e-6) {
    if (!std::isfinite(actual) || std::abs(actual-expected)>tolerance)
        throw std::runtime_error("actual="+std::to_string(actual)+" expected="+std::to_string(expected));
}

template<class T> void gemm_reference() {
    constexpr std::size_t m=2,n=3,k=4;
    for (bool ta: {false,true}) for (bool tb: {false,true}) {
        T a[m*n],b[n*k],c[m*k]{};
        for (std::size_t i=0;i<m;++i) for (std::size_t j=0;j<n;++j)
            a[ta?j*m+i:i*n+j]=T(i+j+1);
        for (std::size_t j=0;j<n;++j) for (std::size_t l=0;l<k;++l)
            b[tb?l*n+j:j*k+l]=T(int(j)-int(l));
        ceras::gemm_cpu(a,ta,b,tb,m,n,k,c);
        const T expected[]={8,2,-4,-10,11,2,-7,-16};
        for(std::size_t i=0;i<m*k;++i) near(c[i],expected[i]);
        if constexpr(ceras::cblas_mode) {
            ceras::cblas_gemm(a,ta,b,tb,m,n,k,c);
            for(std::size_t i=0;i<m*k;++i) near(c[i],expected[i]);
        }
        if constexpr(ceras::cuda_mode) {
            ceras::cuda_gemm(a,ta,b,tb,m,n,k,c);
            for(std::size_t i=0;i<m*k;++i) near(c[i],expected[i]);
        }
    }
}

int main(int argc,char** argv) try {
    require(argc==2,"select a test");
    const std::string mode=argv[1];
    if(mode=="gemm") { gemm_reference<float>(); gemm_reference<double>(); }
    else if(mode=="storage") {
        tensor<double> a({2},1.0), b=a;
        b.resize({3});
        require(a.size()==2 && a.shape()==std::vector<unsigned long>{2},"resize corrupts alias");
        bool rejected=false;
        try { tensor<double> huge({std::numeric_limits<unsigned long>::max()/2+1,2}); }
        catch(const std::exception&) {rejected=true;}
        require(rejected,"shape overflow accepted");
    } else if(mode=="parse") {
        tensor<double> a({1},7.0);
        std::istringstream input("1\n1 2 3\n");
        ceras::read_tensor(input,a);
        require(input.fail(),"oversized input accepted"); near(a[0],7);
    } else if(mode=="image") {
        tensor<unsigned char> a({2,2},static_cast<unsigned char>(37));
        auto b=ceras::imageio::imresize(a,{4,4});
        for(auto x:b) near(x,37,1);
        auto same=resize_other(b);for(auto x:same)near(x,37,1);
        for(auto shape:{std::vector<unsigned long>{2},std::vector<unsigned long>{2,2,5},std::vector<unsigned long>{0,2}}) {
            bool rejected=false;try{ceras::imageio::imresize(tensor<unsigned char>(shape),shape);}catch(std::exception const&){rejected=true;}
            require(rejected,"invalid image dimensions accepted");
        }
        tensor<unsigned char> color({2,3,3},static_cast<unsigned char>(23));
        auto resized=ceras::imageio::imresize(color,{4,6,3});for(auto x:resized)near(x,23,1);
    } else if(mode=="hdr") {
        const auto path=(std::filesystem::temp_directory_path()/"ceras-regression.hdr").string();
        require(ceras::imageio::imwrite(path,tensor<float>({2,2,3},0.5f)),"HDR write failed");
        int w=0,h=0,c=0;
        float* pixels=stbi_loadf(path.c_str(),&w,&h,&c,0);
        require(pixels && w==2 && h==2 && c==3,"HDR dimensions");
        const auto first=pixels[0]; stbi_image_free(pixels); std::filesystem::remove(path);
        near(first,0.5,0.01);
    } else if(mode=="idx") {
        auto dir=std::filesystem::temp_directory_path()/"ceras-regression-idx";
        std::filesystem::create_directories(dir);
        const std::vector<unsigned char> header={0,0,8,3,0,0,0,1,0,0,0,28,0,0,0,28};
        for(auto name:{"train-images-idx3-ubyte","t10k-images-idx3-ubyte"}) {
            std::ofstream out(dir/name,std::ios::binary);
            out.write(reinterpret_cast<char const*>(header.data()),header.size());
            std::vector<char> pixels(785); out.write(pixels.data(),pixels.size());
        }
        for(auto name:{"train-labels-idx1-ubyte","t10k-labels-idx1-ubyte"}) {
            std::ofstream out(dir/name,std::ios::binary);
            const char label[]={0,0,8,1,0,0,0,1,0};out.write(label,sizeof label);
        }
        bool rejected=false;try {ceras::dataset::mnist::load_data(dir.string());}
        catch(std::exception const&){rejected=true;}
        require(rejected,"trailing IDX byte accepted");
        for(auto name:{"train-images-idx3-ubyte","t10k-images-idx3-ubyte"})std::filesystem::resize_file(dir/name,16+784);
        auto valid=ceras::dataset::mnist::load_data(dir.string());require(std::get<0>(valid).size()==784,"valid IDX rejected");
        {std::fstream out(dir/"train-labels-idx1-ubyte",std::ios::binary|std::ios::in|std::ios::out);out.seekp(8);out.put(char(255));}
        rejected=false;try {ceras::dataset::mnist::load_data(dir.string());}catch(std::exception const&){rejected=true;}
        require(rejected,"out of range label accepted");
        std::filesystem::remove_all(dir);
    } else if(mode=="parallel") {
        std::atomic<unsigned long> count{0},outside{0};
        ceras::parallel([&](unsigned long i){++count;if(i<1000||i>=1048)++outside;},1000UL,1048UL,0);
        require(count==48 && outside==0,"parallel range violated");
    } else if(mode=="graph") {
        auto x=ceras::variable{tensor<double>({2},1.0)};
        auto r=ceras::relu(x); auto loss=ceras::sum_reduce(r+r);
        auto out=ceras::get_default_session<tensor<double>>().run(loss);
        loss.backward(ceras::ones_like(out));
        for(auto g:x.gradient()) near(g,2);
    } else if(mode=="softmax") {
        auto x=ceras::variable{tensor<double>({1,2},0.0)};
        auto op=ceras::softmax(x);
        auto out=ceras::get_default_session<tensor<double>>().run(op);
        auto upstream=ceras::ones_like(out); op.backward(upstream);
        for(auto g:x.gradient()) near(g,0);
        for(auto g:upstream) near(g,1);
    } else if(mode=="mean") {
        auto x=ceras::variable{tensor<double>({2,3},1.0)};
        auto op=ceras::mean_reduce(x);
        auto out=ceras::get_default_session<tensor<double>>().run(op);
        op.backward(ceras::ones_like(out));
        for(auto g:x.gradient()) near(g,1.0/6);
    } else if(mode=="elu" || mode=="gelu") {
        auto check=[](auto op,auto& x) {
            auto out=ceras::get_default_session<tensor<double>>().run(op);
            op.backward(ceras::zeros_like(out));
            for(auto g:x.gradient()) near(g,0);
        };
        auto x=ceras::variable{tensor<double>({2},{-1.0,1.0})};
        if(mode=="elu") check(ceras::elu(1.0)(x),x); else check(ceras::gelu(x),x);
    } else if(mode=="maximum") {
        auto a=ceras::max(tensor<double>({1,2},{-2.0,-1.0}),1); near(a[0],-1);
    } else if(mode=="lifetime") {
        auto& s=ceras::get_default_session<tensor<double>>();
        std::weak_ptr<ceras::variable_state<tensor<double>>> weak;
        {auto x=ceras::variable{tensor<double>({1},1.0)}; weak=x.state_;}
        require(weak.expired(),"session retains dead variable");
        auto p=ceras::place_holder<tensor<double>>{};
        for(int i=0;i<2000;++i)s.bind(p,tensor<double>({1},1.0));
        require(s.place_holders_.size()<=1,"duplicate placeholder registrations");
    } else if(mode=="optimizers") {
        auto x=ceras::variable{tensor<double>({1},2.0)};
        auto unrelated=ceras::variable{tensor<double>({1},9.0)};
        auto loss=ceras::sum_reduce(ceras::square(x));
        auto& s=ceras::get_default_session<tensor<double>>();
        auto opt=ceras::sgd{loss,1UL,0.1,0.0,0.0,false};
        s.run(loss);s.run(opt);near(x.data()[0],1.6);near(unrelated.data()[0],9);
        auto adam=ceras::adam{loss,1UL,0.1};
        s.run(loss);s.run(adam);near(x.data()[0],1.5,1e-5);
    } else if(mode=="adadelta") {
        auto x=ceras::variable{tensor<double>({1},2.0)};
        auto loss=ceras::sum_reduce(ceras::square(x));
        auto opt=ceras::adadelta{loss,1UL,0.9};
        auto& s=ceras::get_default_session<tensor<double>>();
        s.run(loss);s.run(opt);
        near(x.data()[0],2.0-4.0*std::sqrt(ceras::eps/(1.6+ceras::eps)),1e-7);
    } else if(mode=="crossentropy") {
        auto labels=ceras::constant{tensor<double>({2,2},{1.0,0.0,1.0,0.0})};
        auto x=ceras::variable{tensor<double>({2,2},0.0)};
        auto loss=ceras::cross_entropy_loss(labels,x);
        auto out=ceras::get_default_session<tensor<double>>().run(loss);
        loss.backward(ceras::ones_like(out));
        near(x.gradient()[0],-0.25);near(x.gradient()[1],0.25);
    } else throw std::runtime_error("unknown test");
    std::cout<<mode<<": PASS\n";
    return 0;
} catch(const std::exception& e) {
    std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1;
}
