#include "../include/utils/float32.hpp"
#include "../include/utils/string.hpp"

template< ceras::float32 F="1.0" >
struct vx
{
    static constexpr float val = F;

    void operator()() const noexcept
    {
        std::cout << val << std::endl;
    }
};



int main()
{
    using namespace ceras;

    vx<>{}();
    vx<"-1.0">{}();
    vx<"1.0">{}();
    vx<"+1.0e1">{}();
    vx<"1.0e-1">{}();
    vx<"-1.0e1">{}();
    vx<"-1.0e-1">{}();

    return 0;
}
