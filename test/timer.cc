#include "../include/utils/timer.hpp"

using namespace ceras;

double result;

auto loop( unsigned long n )
{
    double x = n*n;
    for ( auto idx = 0; idx != n; ++idx )
    {
        x = std::cos(x);
    }
    result = x;
}

int main()
{
    unsigned long tests = 1024;
    for ( auto n = 0; n != tests; ++n )
    {
        time_it( [n](){ loop(n+1); } );
    }

    std::cout << "not important result: " << result << std::endl;
    return 0;
}

