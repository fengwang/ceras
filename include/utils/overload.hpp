#ifndef OVERLOAD_HPP_INCLUDED_ASDKALSFKDJASODSIJALSKJASLKJASLKSJDAAAAAAKKFJIEKDA
#define OVERLOAD_HPP_INCLUDED_ASDKALSFKDJASODSIJALSKJASLKJASLKSJDAAAAAAKKFJIEKDA

namespace ceras
{
    template<class... Ts>
    struct overload : Ts...
    {
        using Ts::operator()...;
    };

    template<class... Ts> overload(Ts...) -> overload<Ts...>;
}

#endif//OVERLOAD_HPP_INCLUDED_ASDKALSFKDJASODSIJALSKJASLKJASLKSJDAAAAAAKKFJIEKDA

