#ifndef XNRPSJMCYFXBDGNRJAWDNDIYQNGNXMRVLEHGNQWILKMTHGNOVHODLLXCCNIMUUFQSMOIYHDUD
#define XNRPSJMCYFXBDGNRJAWDNDIYQNGNXMRVLEHGNQWILKMTHGNOVHODLLXCCNIMUUFQSMOIYHDUD

#include "./operation.hpp"
#include "./place_holder.hpp"
#include "./variable.hpp"
#include "./session.hpp"
#include "./utils/color.hpp"
#include "./utils/debug.hpp"

namespace ceras
{
    // Example usage:
    //
    //  session ss;
    //  auto loss = ...;
    //  auto optimizer = gradient{ loss, 1.0e-3f };
    //  for i = 1 : 1000
    //      ss.run( loss, batch_size )
    //      ss.run( optimizer )
    //
    template< typename Loss, typename T >
    struct gradient_descent
    {
        Loss& loss_;
        T learning_rate_;
        T momentum_;

        gradient_descent(Loss& loss, std::size_t batch_size, T learning_rate=1.0e-3, T momentum=0.0) noexcept : loss_(loss), learning_rate_(learning_rate), momentum_(momentum)
        {
            learning_rate_ /= static_cast<T>( batch_size ); // fix for batch size
        }

        void forward()
        {
            // update the gradient in the loss
            loss_.backward( ones<T>({1,}) );
            //update variables
            auto& ss = get_default_session<T,default_allocator<T>>().get();
            for ( auto [id, v] : ss.variables_ )
            {
                *(v.get().data_) -= learning_rate_ * (*(v.get().gradient_)) * (1.0-momentum_);
                *(v.get().data_) -= learning_rate_ * (*(v.get().old_gradient_)) * momentum_;
            }
        }

    };

}//namespace ceras

#endif//XNRPSJMCYFXBDGNRJAWDNDIYQNGNXMRVLEHGNQWILKMTHGNOVHODLLXCCNIMUUFQSMOIYHDUD

