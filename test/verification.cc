#include "ceras.hpp"
#include <iostream>
#include <stdexcept>

using ceras::tensor;
void check(bool b) {if(!b)throw std::runtime_error("verification failed");}
void close(double a,double b,double tol=1e-5) {
    if(!std::isfinite(a)||std::abs(a-b)>tol*(1+std::abs(b)))
        throw std::runtime_error("actual="+std::to_string(a)+" expected="+std::to_string(b));
}
template<class F> void throws(F f) {
    bool caught=false;try{f();}catch(std::exception const&){caught=true;}check(caught);
}
template<class Builder> void gradient(Builder build) {
    auto x=ceras::variable{tensor<double>({2,3},{-1.2,-0.3,0.2,0.7,1.1,2.0})};
    auto op=build(x);std::cout<<"checking "<<op.name()<<std::endl;auto& s=ceras::get_default_session<tensor<double>>();
    auto y=s.run(op);auto upstream=ceras::ones_like(y);
    for(std::size_t i=0;i<upstream.size();++i)upstream[i]=0.3+i*0.2;
    op.backward(upstream);auto analytic=x.gradient().deep_copy();
    constexpr double h=1e-6;
    for(std::size_t i=0;i<x.data().size();++i) {
        double old=x.data()[i];
        x.data()[i]=old+h;auto plus=s.run(op).deep_copy();
        x.data()[i]=old-h;auto minus=s.run(op).deep_copy();
        x.data()[i]=old;
        double numeric=0;
        for(std::size_t j=0;j<plus.size();++j)numeric+=(plus[j]-minus[j])*upstream[j]/(2*h);
        close(analytic[i],numeric);
    }
}
template<class T> struct failing_allocator : std::allocator<T> {
    using value_type=T;
    static inline bool fail=false;
    template<class U> struct rebind {using other=failing_allocator<U>;};
    T* allocate(std::size_t n) {
        if(fail)throw std::bad_alloc();return std::allocator<T>::allocate(n);
    }
};
int main(int argc,char** argv) try {
    check(argc==2);std::string mode=argv[1];
    if(mode=="gradients") {
        gradient([](auto& x){return ceras::softmax(x);});
        gradient([](auto& x){return ceras::mean_reduce(x);});
        gradient([](auto& x){return ceras::elu(0.7)(x);});
        gradient([](auto& x){return ceras::gelu(x);});
        gradient([](auto& x){return ceras::sigmoid(x);});
        gradient([](auto& x){return ceras::square(x);});
        gradient([](auto& x){auto labels=ceras::constant{tensor<double>({2,3},{1.,0.,0.,0.,1.,0.})};return ceras::cross_entropy_loss(labels,x);});
        gradient([](auto& x){auto r=ceras::relu(x);return ceras::sum_reduce(r+ceras::square(r));});
    } else if(mode=="boundaries") {
        throws([]{tensor<double> bad({2},{1.0});});
        tensor<double> x({2,3},1.0);
        throws([&]{x.reshape({0,~0UL});});
        throws([&]{x.reshape({4,~0UL});});
        throws([&]{x.reshape({7});});
        throws([&]{(void)ceras::mean(x,2);});
        throws([&]{(void)ceras::max(x,3);});
        throws([]{(void)ceras::max(tensor<double>{});});
        auto inf=std::numeric_limits<double>::infinity();
        auto nan=std::numeric_limits<double>::quiet_NaN();
        check(ceras::max(tensor<double>({2},-inf))==-inf);
        check(ceras::max(tensor<double>({1,2},-inf),1)[0]==-inf);
        check(std::isnan(ceras::max(tensor<double>({2},{nan,1.0}))));
        check(std::isnan(ceras::max(tensor<double>({1,2},{nan,1.0}),1)[0]));
        close(ceras::sum(tensor<double>({2},1.0),0)[0],2);
        auto original=x.deep_copy();auto ignored=x+x;
        for(std::size_t i=0;i<x.size();++i)close(x[i],original[i]);
    } else if(mode=="contexts") {
        ceras::ceras_private::session<tensor<double>> a,b;
        a.seed_random(42);b.seed_random(42);
        auto draw=[]{return ceras::random<double>({2,3});};
        auto first=a.generate(draw),second=b.generate(draw);
        for(std::size_t i=0;i<first.size();++i)close(first[i],second[i]);
        a.generate(draw);b.seed_random(42);auto again=b.generate(draw);
        for(std::size_t i=0;i<first.size();++i)close(first[i],again[i]);
        auto x=ceras::variable{tensor<double>({2},{-1.0,2.0})};
        auto op=ceras::relu(x);
        a.run(op);b.run(op);a.backward(op,tensor<double>({2},1.0));close(x.gradient()[1],1);
        a.clear_forward_cache();throws([&]{a.backward(op,tensor<double>({2},1.0));});
        auto old=ceras::learning_phase;
        try{ceras::learning_phase_scope scope{0};throw std::runtime_error("test");}catch(...){}
        check(ceras::learning_phase==old);
        std::atomic<int> completed=0;
        std::vector<std::jthread> threads;
        for(int t=0;t<4;++t) threads.emplace_back([&]{
            ceras::ceras_private::session<tensor<double>> context;
            auto variable=ceras::variable{tensor<double>({2},2.0)};
            auto expression=ceras::sum_reduce(ceras::square(ceras::relu(variable)));
            for(int step=0;step<20;++step){context.run(expression);context.backward(expression,tensor<double>({1},1.0));close(variable.gradient()[0],4);}
            ++completed;
        });
        threads.clear();check(completed==4);
        auto rows=ceras::softmax(tensor<double>({2048,8},1.0));
        for(auto value:rows)close(value,0.125,1e-12);
    } else if(mode=="recurrences") {
        auto exercise=[](auto factory,auto update) {
            auto x=ceras::variable{tensor<double>({1},2.0)};
            auto loss=ceras::sum_reduce(ceras::square(x));auto opt=factory(loss);
            auto& session=ceras::get_default_session<tensor<double>>();
            double reference=2.0;
            for(int step=1;step<=5;++step) {
                reference=update(reference,2*reference,step);
                session.run(loss);session.run(opt);close(x.data()[0],reference,1e-7);
            }
        };
        for(bool nesterov:{false,true})
            exercise([=](auto& l){return ceras::sgd{l,1UL,0.1,0.6,0.0,nesterov};},
                     [=,m=0.0](double x,double g,int)mutable{m=0.6*m-0.1*g;return x+(nesterov?0.6*m-0.1*g:m);});
        exercise([](auto& l){return ceras::adagrad{l,1UL,0.1};},
                 [v=0.0](double x,double g,int)mutable{v+=g*g;return x-0.1*g/(std::sqrt(v)+ceras::eps);});
        exercise([](auto& l){return ceras::rmsprop{l,1UL,0.1,0.9};},
                 [v=0.0](double x,double g,int)mutable{v=0.9*v+0.1*g*g;return x-0.1*g/(std::sqrt(v)+ceras::eps);});
        exercise([](auto& l){return ceras::adadelta{l,1UL,0.9};},
                 [v=0.0,d=0.0](double x,double g,int)mutable{v=0.9*v+0.1*g*g;double step=g*std::sqrt((d+ceras::eps)/(v+ceras::eps));d=0.9*d+0.1*step*step;return x-step;});
        exercise([](auto& l){return ceras::adam{l,1UL,0.1,0.9,0.999};},
                 [m=0.0,v=0.0](double x,double g,int t)mutable{m=0.9*m+0.1*g;v=0.999*v+0.001*g*g;return x-0.1*(m/(1-std::pow(0.9,t)))/(std::sqrt(v/(1-std::pow(0.999,t)))+ceras::eps);});
    } else if(mode=="optimizer_state") {
        auto x=ceras::variable{tensor<double>({1},2.0)};
        auto loss=ceras::sum_reduce(ceras::square(x));auto opt=ceras::adam{loss,1UL,0.1};
        auto& s=ceras::get_default_session<tensor<double>>();s.run(loss);s.run(opt);
        auto copy=opt;
        check(copy.states_.at(x.id()).first.data()!=opt.states_.at(x.id()).first.data());
        opt.reset_state();check(opt.states_.empty());
        x.data().resize({2});x.data().reset(2.0);s.run(loss);s.run(opt);
        for(auto v:x.data())close(v,1.9,1e-7);
        x.trainable(false);s.run(loss);s.run(opt);for(auto v:x.data())close(v,1.9,1e-7);
        throws([&]{s.run(opt);});
    } else if(mode=="normalization") {
        auto exercise=[](unsigned long batch) {
            auto x=ceras::variable{tensor<double>({1,2},{0.2,-0.3})};
            auto rows=ceras::constant{tensor<double>({batch,1},1.0)};
            tensor<double> y({batch,2});for(unsigned long i=0;i<batch;++i)y[2*i]=1;
            auto labels=ceras::constant{y};auto loss=ceras::cross_entropy_loss(labels,rows*x);
            auto opt=ceras::sgd{loss,batch,0.1};auto& session=ceras::get_default_session<tensor<double>>();
            auto before=session.run(loss)[0];session.run(opt);
            return std::array<double,3>{before,x.data()[0],x.data()[1]};
        };
        auto once=exercise(1),twice=exercise(2);
        for(unsigned i=0;i<3;++i)close(once[i],twice[i],1e-10);
    } else if(mode=="training") {
        ceras::seed_random(42);
        auto x=ceras::variable{tensor<double>({2},{-2.0,3.0})};
        auto target=ceras::constant{tensor<double>({2},{1.0,-1.0})};
        auto loss=ceras::mean_reduce(ceras::square(x-target));auto opt=ceras::sgd{loss,2UL,0.1};
        auto& session=ceras::get_default_session<tensor<double>>();
        double initial=session.run(loss)[0];
        for(int i=0;i<100;++i){session.run(loss);session.run(opt);}
        check(session.run(loss)[0]<initial*1e-8);
    } else if(mode=="allocation") {
        check(tensor<double>{}.deep_copy().empty());
        using A=failing_allocator<double>;using T=tensor<double,A>;
        T value({2},7.0);
        A::fail=true;throws([&]{value.resize({3});});A::fail=false;
        check(value.size()==2);close(value[0],7);
        A::fail=true;throws([&]{auto ignored=ceras::deep_copy(value);});A::fail=false;
        throws([]{tensor<double> huge({std::numeric_limits<unsigned long>::max()/sizeof(double)+1});});
        throws([]{(void)ceras::checked_backend_dimension(std::size_t(INT_MAX)+1);});
        struct alignas(128) aligned {int value;};
        ceras::cached_allocator<aligned> allocator;
        auto ptr=allocator.allocate(2);check(reinterpret_cast<std::uintptr_t>(ptr)%128==0);allocator.deallocate(ptr,2);
        ceras::vector<double,std::allocator<double>> a(2,7.0),b(3,9.0);
        a=std::move(b);check(a.size()==3);close(a.data()[0],9);
        a=std::move(a);a.resize(4);check(a.size()==4);
    } else if(mode=="restore") {
        auto x=ceras::variable{tensor<double>({1},3.0)};
        auto y=ceras::variable{tensor<double>({1},4.0)};
        auto& s=ceras::get_default_session<tensor<double>>();
        std::ostringstream text;s.write_original(text);
        x.data()[0]=9;y.data()[0]=10;
        std::istringstream valid(text.str());s.read_original(valid);close(x.data()[0],3);close(y.data()[0],4);
        auto bad=std::to_string(x.id())+" "+std::to_string(y.id())+"\n1\n20\n1\n1 2\n";
        std::istringstream input(bad);throws([&]{s.read_original(input);});close(x.data()[0],3);close(y.data()[0],4);
        std::istringstream unknown("-123\n1\n2\n");throws([&]{s.read_original(unknown);});
        auto path=std::filesystem::temp_directory_path()/"ceras-session-regression.bin";
        s.save(path.string());x.data()[0]=8;s.restore(path.string());close(x.data()[0],3);std::filesystem::remove(path);
    } else if(mode=="ranges") {
        std::atomic<int> near_limit=0;
        ceras::parallel([&](unsigned long){++near_limit;},~0UL-48,~0UL,0,8);check(near_limit==48);
        for(unsigned workers:{0,1,2,8})for(unsigned n:{0,1,2,17,48}) {
            std::vector<std::atomic<int>> visited(n);
            ceras::parallel([&](unsigned i){check(i>=100&&i<100+n);++visited[i-100];},100U,100U+n,0,workers);
            for(auto& v:visited)check(v==1);
        }
        throws([]{ceras::parallel([](unsigned){},5U,4U);});
        throws([]{ceras::parallel([](unsigned i){if(i==7)throw std::runtime_error("callback");},0U,20U,0,4);});
    } else if(mode=="parsers") {
        auto& limits=ceras::tensor_io_limits;auto old=limits;limits.max_elements=32;limits.max_line_bytes=128;
        for(auto text:{"1\n1 2\n","-1\n0\n","99999\n0\n","1x\n0\n","1\nbad\n","1 2\n1\n","\n\n"}) {
            tensor<double> value({1},7.0);std::istringstream in(text);ceras::read_tensor(in,value);check(in.fail());close(value[0],7);
        }
        std::mt19937 rng(123);
        int rounds=2000;
        if(auto value=std::getenv("CERAS_FUZZ_ROUNDS"))rounds=std::clamp(std::stoi(value),1,1000000);
        for(int round=0;round<rounds;++round) {
            std::string bytes;for(unsigned i=0,n=rng()%100;i<n;++i)bytes+=char(rng()%128);
            std::istringstream in(bytes);tensor<double> value;ceras::read_tensor(in,value);
            std::istringstream compressed(bytes);std::ostringstream out;
            (void)lzw::decompress(compressed,out,512);check(out.str().size()<=512);
        }
        limits=old;
    } else throw std::runtime_error("unknown verification");
    std::cout<<mode<<": PASS\n";return 0;
} catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
