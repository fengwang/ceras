#include "ceras.hpp"
#include <atomic>
#include <iostream>
int main() {
    if constexpr (!ceras::cuda_mode) return 77;
    ceras::cuda_backend_context shared(0);
    std::atomic<bool> failed{false};
    std::vector<std::jthread> threads;
    for(int worker=0;worker<4;++worker) threads.emplace_back([&,worker] {
        try {
            for(int pass=0;pass<20;++pass) {
                std::size_t n=8+worker*7+pass;
                std::vector<double> a(n*n,worker+1), b(n*n,2), c(n*n);
                if(pass%2) ceras::cuda_gemm(a.data(),false,b.data(),false,n,n,n,c.data(),shared);
                else ceras::cuda_gemm(a.data(),false,b.data(),false,n,n,n,c.data());
                for(auto x:c) if(std::abs(x-2*n*(worker+1))>1e-7) failed=true;
            }
        } catch(std::exception const& e) {std::cerr<<e.what()<<'\n';failed=true;}
    });
    threads.clear();
    return failed ? 1 : 0;
}
