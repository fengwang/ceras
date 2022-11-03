#include "../../include/utils/buffered_allocator.hpp"
#include "../../include/utils/range.hpp"

TEST_CASE( "buffered_allocator", "[buffered_allocator_1]" )
{

    for ( int start = -100; start < 100; ++start )
    {
        for ( int n = 5; n < 200; ++n )
        {
            {
                std::vector<int> v;
                v.reserve( n );

                for ( int i = start; i < n+start; ++i )
                    v.push_back( i );

                std::vector<int, ceras::buffered_allocator<int, 128> > u{ v.begin(), v.end() };

                for ( auto idx : ceras::range(n) )
                     REQUIRE( u[idx] == v[idx] );
            }

            {
                std::vector<int, ceras::buffered_allocator<int, 128> > v;
                v.reserve( n );

                for ( int i = start; i < n+start; ++i )
                    v.push_back( i );

                std::vector<int> u{ v.begin(), v.end() };

                for ( auto idx : ceras::range(n) )
                     REQUIRE( u[idx] == v[idx] );
            }

        }
    }

}

