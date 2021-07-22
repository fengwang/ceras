#include "../include/utils/state.hpp"

struct st
{
    char c_;
    int i_;
    float f_;
};

struct h : ceras::enable_shared_state<h, st>
{
    int x_;
};

int main()
{
    h h1;
    auto h2 = h1;

    (*(h1.state_)).i_ = 12;
    assert( (*(h2.state_)).i_ == 12 );

    (*(h2.state_)).i_ = 112;
    assert( (*(h1.state_)).i_ == 112 );

    return 0;
}

