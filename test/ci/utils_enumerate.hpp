#include "../../include/utils/enumerate.hpp"

TEST_CASE( "enumerate", "[enumerate_1]" )
{

    for ( int start = -100; start < 100; ++start )
    {
        for ( int n = 5; n < 200; ++n )
        {
            std::vector<int> v;
            v.reserve( n );

            for ( int i = start; i < n+start; ++i )
                v.push_back( i );

            for ( auto const& [idx, val] : ceras::enumerate( v ) )
                REQUIRE( static_cast<long>(val) == static_cast<long>(idx) + static_cast<long>(start) );
        }
    }

}

