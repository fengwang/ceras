#include "../include/utils/for_each.hpp"
#include "../include/utils/timer.hpp"
#include "../include/utils/fmt.hpp"

#include <vector>

void test_1()
{
    unsigned long const n = 200000000;
    std::vector<float> va(n, 0.1f);

    {
        ceras::timer t{ fmt::format( "vector modification with {} elements", n) };
        ceras::for_each( va.begin(), va.end(), []( auto& v ){ v += -1.0f; } );
    }

}

int main()
{
    test_1();
    test_1();

    return 0;
}

