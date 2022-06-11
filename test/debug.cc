#include "../include/utils/debug.hpp"

int main()
{
    ceras::debug_log( "start debugging." );
    ceras::debug_info( "debug info." );
    ceras::debug_error( "debug error." );
    ceras::debug_warn( "debug warn." );
    ceras::debug_critical( "debug critical." );

    return 0;
}

