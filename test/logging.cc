#include "../include/utils/logging.hpp"


void test( int level )
{
    std::cout << "\ntesting logging at level " << level << "\n\n";
    logging::set_level( level );
    logging::log( "log at level {}.", level );
    logging::debug( "debug at level {}.", level );
    logging::info( "info at level {}.", level );
    logging::warning( "warning at level {}.", level );
    logging::error( "error at level {}.", level );
    logging::critical( "critical at level {}.", level );
}

int main()
{
    test( -5 );
    test( 5 );
    test( 15 );
    test( 25 );
    test( 35 );
    test( 45 );
    test( 55 );

    return 0;
}

