#ifndef XRPQQWRCMKHXJGYGUOKUFVOOOAGKKIFAPYKPWPWNUITTIQTMANSLPOHWBPFAACFYPYEOWIMIN
#define XRPQQWRCMKHXJGYGUOKUFVOOOAGKKIFAPYKPWPWNUITTIQTMANSLPOHWBPFAACFYPYEOWIMIN

#include "../includes.hpp"

namespace ceras
{
    // adding state to host class
    //
    // struct st{ int a, char c };
    //
    // struct v : enable_shared_state<v, st>
    // {
    // };
    //
    // v v1;
    // v v2 = v1; // <-- now v1 and v2 share same state
    // *(v1.state_).a = 1; // <-- (v2.state_).a updated as well
    //
    template<typename Host, typename State>
    struct enable_shared_state
    {
        std::shared_ptr<State> state_;

        enable_shared_state() : state_{ std::make_shared<State>() } {}
    };

}//namespace ceras

#endif//XRPQQWRCMKHXJGYGUOKUFVOOOAGKKIFAPYKPWPWNUITTIQTMANSLPOHWBPFAACFYPYEOWIMIN

