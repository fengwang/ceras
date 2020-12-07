#ifndef QUMRFMABNSVNVMCQHSCMKQHSXIJUKQKIXLUYRKNOYDQPQPWJTUYXCPTIVAWGTONYSSVFULTWY
#define QUMRFMABNSVNVMCQHSCMKQHSXIJUKQKIXLUYRKNOYDQPQPWJTUYXCPTIVAWGTONYSSVFULTWY


    template< typename T, typename A >
    std::ostream& operator << ( std::ostream& os, std::vector<T, A> const& vec )
    {
        os << "[  ";
        std::copy( vec.begin(), vec.end(), std::ostream_iterator<T>{os, "  "} );
        os << "]";
        return os;
    }


#endif//QUMRFMABNSVNVMCQHSCMKQHSXIJUKQKIXLUYRKNOYDQPQPWJTUYXCPTIVAWGTONYSSVFULTWY

