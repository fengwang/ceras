#ifndef GIUMPNTRSJLBBLAUORAVISIMRPYVDLKYQMYYNMRNXKVNTYBNXKWVRSEGRINPAGYELAFMRASAT
#define GIUMPNTRSJLBBLAUORAVISIMRPYVDLKYQMYYNMRNXKVNTYBNXKWVRSEGRINPAGYELAFMRASAT

#include "../includes.hpp"

namespace ceras
{

    template< typename Concrete_Type >
    struct enable_shared: std::enable_shared_from_this<Concrete_Type>
    {
        std::shared_ptr<Concrete_Type> shared()
        {
            auto& zen = static_cast<Concrete_Type&>(*this);
            return zen.shared_from_this();
        };

        std::shared_ptr<Concrete_Type> shared() const
        {
            auto const& zen = static_cast<Concrete_Type const&>(*this);
            return zen.shared_from_this();
        };
    };

}//namespace ceras

#endif//GIUMPNTRSJLBBLAUORAVISIMRPYVDLKYQMYYNMRNXKVNTYBNXKWVRSEGRINPAGYELAFMRASAT

