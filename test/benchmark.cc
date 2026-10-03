#include "ceras.hpp"
#include <sys/resource.h>
#include <atomic>
#include <cstdlib>
#include <new>
static std::atomic<std::size_t> allocations{0};
void* operator new(std::size_t n) {++allocations;if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
template<class T> void run() {
    for(unsigned long rows:{1UL,8UL})for(unsigned long width:{7UL,8UL,9UL,32UL,static_cast<unsigned long>(std::max(1U,std::thread::hardware_concurrency()))}) {
        ceras::tensor<T> x({rows,width},T{0.1});
        std::vector<double> timings;std::size_t threads=0;
        ceras::parallel_thread_counter=&threads;
        auto start_allocations=allocations.load();double cold=0;
        for(int sample=0;sample<6;++sample) {
            auto start=std::chrono::steady_clock::now();auto y=ceras::softmax(x);
            auto end=std::chrono::steady_clock::now();
            if(std::abs(ceras::sum(y)-rows)>0.001)throw std::runtime_error("softmax parity");
            double us=std::chrono::duration<double,std::micro>(end-start).count();
            if(sample==0)cold=us;else timings.push_back(us);
        }
        ceras::parallel_thread_counter=nullptr;
        auto count=allocations.load()-start_allocations;
        std::sort(timings.begin(),timings.end());rusage usage{};getrusage(RUSAGE_SELF,&usage);
        std::cout<<sizeof(T)<<','<<rows<<','<<width<<','<<cold<<','<<timings.front()<<','<<timings[2]<<','<<timings.back()<<','<<threads<<','<<count<<','<<usage.ru_maxrss<<'\n';
    }
}
int main(){std::cout<<"bytes,rows,width,cold_us,min_us,median_us,max_us,threads,ordinary_new_calls,process_peak_rss_kib\n";run<float>();run<double>();}
