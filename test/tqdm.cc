#include "../include/utils/tqdm.hpp"
#include <thread>
#include <vector>

int main()
{
    {
        std::vector<int> A = {1, 2, 3, 4, 5, 6, 7 };
        for ( [[maybe_unused]]int a : tq::dm( A ) )
            std::this_thread::sleep_for( std::chrono::milliseconds( 765 ) );
    }
    {
        std::vector<int> A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
        for ( [[maybe_unused]]int a : tq::dm( A ) )
            std::this_thread::sleep_for( std::chrono::milliseconds( 765 ) );
    }

    {
        std::vector<int> A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
        for ( [[maybe_unused]]int a : tq::dm( A ) )
            std::this_thread::sleep_for( std::chrono::milliseconds( 567 ) );
    }

    {
        for ( [[maybe_unused]]int a : tq::trange( 30 ) )
            std::this_thread::sleep_for( std::chrono::milliseconds( 467 ) );
    }

    return 0;
}

