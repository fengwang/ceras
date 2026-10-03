#include "ceras.hpp"
#include <stdexcept>
#include <iostream>
void need(bool value) {if(!value)throw std::runtime_error("API regression");}
int main() {
    using ceras::tensor;
    auto p=ceras::place_holder<tensor<float>>{};
    ceras::bind(p,tensor<float>({1},2.0f));need(ceras::run(p)[0]==2);
    ceras::get_default_session<tensor<float>>().tap(p);
    auto weight=ceras::variable{tensor<float>({1},1.0f)};
    auto out=ceras::relu(ceras::hadamard_product(p,weight));
    auto model=ceras::model{p,out};
    model.trainable(false);need(!weight.trainable());
    model.trainable(true);need(weight.trainable());
    auto replacement=ceras::place_holder<tensor<float>>{};
    auto composed=model(replacement);
    ceras::bind(replacement,tensor<float>({1},3.0f));
    need(ceras::run(composed)[0]==3);
    auto make_compiled=[&] {return model.compile(ceras::MeanSquaredError(),ceras::SGD(1UL,0.1f));};
    auto moved=[&]{auto source=make_compiled();return std::move(source);}();
    moved.train_on_batch(tensor<float>({1},1.0f),tensor<float>({1},0.0f));
    need(weight.data()[0]<1.0f);
    moved.trainable(false);
    auto before=weight.data()[0];
    moved.train_on_batch(tensor<float>({1},1.0f),tensor<float>({1},0.0f));
    need(weight.data()[0]==before);
    auto delayed=ceras::concatenate(ceras::relu(weight),ceras::relu(weight));
    auto joined=delayed(0);need(ceras::run(joined).size()==2);
    std::cout<<"API PASS\n";
}
