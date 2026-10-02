#pragma once
#include "../includes.hpp"
#include "../config.hpp"
namespace ceras {
inline thread_local std::size_t* parallel_thread_counter=nullptr;
inline thread_local unsigned long parallel_min_work=4096;
inline thread_local bool inside_parallel=false;
template<class Function, std::unsigned_integral I>
void parallel(Function const& func, I first, I last, unsigned long threshold=parallel_min_work,
              unsigned int workers=std::thread::hardware_concurrency()) {
    if(last<first) throw std::invalid_argument("reversed parallel range");
    auto n=last-first;
    if(!parallel_mode || inside_parallel || n<=threshold || workers<=1) {
        for(auto i=first;i<last;++i) func(i);
        return;
    }
    workers=static_cast<unsigned int>(std::min<std::uintmax_t>(workers, threshold ? std::max<std::uintmax_t>(1,n/threshold) : n));
    std::exception_ptr failure;
    std::mutex error_mutex;
    auto run=[&](I a,I b) {
        struct nesting {bool previous=inside_parallel;nesting(){inside_parallel=true;}~nesting(){inside_parallel=previous;}} scope;
        try {for(auto i=a;i<b;++i) func(i);}
        catch(...) {std::lock_guard lock(error_mutex);if(!failure) failure=std::current_exception();}
    };
    {
        std::vector<std::jthread> threads;
        threads.reserve(workers-1);
        I cursor=first, base=n/workers, extra=n%workers;
        for(unsigned int w=0;w<workers;++w) {
            I end=cursor+base+(w<extra?1:0);
            if(w+1==workers) run(cursor,end);
            else {threads.emplace_back(run,cursor,end);if(parallel_thread_counter) ++*parallel_thread_counter;}
            cursor=end;
        }
    }
    if(failure) std::rethrow_exception(failure);
}
template<class Function,class I>
void parallel(Function const& f,I last) {parallel(f,I{0},last);}
}
