#include <version>
#include <iostream>
int main(){
    std::cout<<"__cplusplus="<<__cplusplus<<'\n';
#ifdef __cpp_lib_span
    std::cout<<"span="<<__cpp_lib_span<<'\n';
#endif
#ifdef __cpp_lib_expected
    std::cout<<"expected="<<__cpp_lib_expected<<'\n';
#endif
#ifdef __cpp_lib_mdspan
    std::cout<<"mdspan="<<__cpp_lib_mdspan<<'\n';
#endif
#ifdef __cpp_lib_inplace_vector
    std::cout<<"inplace_vector="<<__cpp_lib_inplace_vector<<'\n';
#endif
#ifdef __cpp_contracts
    std::cout<<"contracts="<<__cpp_contracts<<'\n';
#endif
}
