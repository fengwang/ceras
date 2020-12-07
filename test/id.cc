#include "../include/utils/id.hpp"
#include <cassert>

int main()
{
    int N = 1000;
    for ( int idx = 0; idx != N; ++idx )
    {
        int g_id = ceras::generate_uid();
        assert( g_id == idx );
    }

    return 0;
}


