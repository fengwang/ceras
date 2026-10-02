#include "ceras.hpp"
#include <iostream>
int main(int argc,char** argv) try {
    if(argc!=2)throw std::runtime_error("MNIST directory required");
    using namespace ceras;seed_random(42);parallel_min_work=65536;
    auto [images,labels,test_images,test_labels]=dataset::mnist::load_data(argv[1]);
    constexpr unsigned long batch=32,train=512,validation=256;
    if(images.shape()[0]<train || test_images.shape()[0]<validation)throw std::runtime_error("insufficient MNIST samples");
    place_holder<tensor<float>> x,y;
    auto weights=variable{tensor<float>({784,10},0.0f)};
    auto logits=x*weights;auto loss=cross_entropy_loss(y,logits);
    auto opt=sgd{loss,batch,0.2f};auto& s=get_default_session<tensor<float>>();
    tensor<float> bx({batch,784}),by({batch,10});s.bind(x,bx);s.bind(y,by);
    auto feed=[&](unsigned long offset,auto const& source,auto const& target){
        for(unsigned long i=0;i<bx.size();++i)bx[i]=source[offset*784+i]/255.0f;
        for(unsigned long i=0;i<by.size();++i)by[i]=target[offset*10+i];
    };
    feed(0,images,labels);float before=s.run(loss)[0];
    for(unsigned long step=0;step<320;++step){feed((step%(train/batch))*batch,images,labels);s.run(loss);s.run(opt);}
    feed(0,images,labels);float after=s.run(loss)[0];unsigned correct=0;
    for(unsigned long offset=0;offset<validation;offset+=batch){
        feed(offset,test_images,test_labels);auto scores=s.run(logits);
        for(unsigned long row=0;row<batch;++row){auto first=scores.data()+row*10;auto label=std::max_element(first,first+10)-first;correct+=by[row*10+label]>0;}
    }
    double accuracy=double(correct)/validation;
    std::cout<<"MNIST seed=42 loss "<<before<<" -> "<<after<<" validation accuracy="<<accuracy<<'\n';
    if(!(after<0.5f*before && accuracy>=0.65))throw std::runtime_error("MNIST convergence gate");
} catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
