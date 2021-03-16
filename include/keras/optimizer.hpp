#ifndef UNNVJXQUIHOGAUQDARAHWUAYFHCTQMPOMMMUELKRAGSVIPSEIQKAUSTGABINSFKPHINELBKVJ
#define UNNVJXQUIHOGAUQDARAHWUAYFHCTQMPOMMMUELKRAGSVIPSEIQKAUSTGABINSFKPHINELBKVJ

#include "../utils/string.hpp"
#include "../utils/float32.hpp"
#include "../optimizer.hpp"
#include "../utils/better_assert.hpp"

namespace Keras
{

    // usage:
    //
    //     auto opt = optimizer< "sgd", 32, "0.1", "0.0", "0.0", true >{};
    //     auto opt = optimizer< "adagrad", 32, "0.1" >{};
    //     auto optim = opt( loss, 32 ); // <- create an instance of an optimizer
    //

    template< ceras::string Name="random_normal", unsigned long batch_size=32, ceras::float32 Arg1="0.1", ceras::float32 Arg2 ="0.0", ceras::float32 Arg3="0.0", bool flag=false >
    struct optimizer
    {
        //static constexpr char const * name_ = Name;
        /*
        static constexpr unsigned long batch_size_ = batch_size;
        static constexpr float arg1_ = Arg1; // mostly learning rate
        static constexpr float arg2_ = Arg2; // momentum
        static constexpr float arg3_ = Arg3; // decay
        static constexpr bool flag_ = flag;  // nesterov flag
        */
        std::string name_ = Name;
        unsigned long batch_size_ = batch_size;
        float arg1_ = Arg1; // mostly learning rate
        float arg2_ = Arg2; // momentum
        float arg3_ = Arg3; // decay
        bool flag_ = flag;  // nesterov flag

        // generate an instance of an optimizer
        template< typename Loss >
        auto operator()( Loss& loss ) const noexcept
        {
            constexpr char const * static_name = Name;

            if constexpr ( ceras::string_equal( static_name, "sgd" ) || ceras::string_equal( static_name, "SGD" ) )
            {
                return ceras::sgd{ loss, batch_size_, arg1_, arg2_, arg3_, flag_ }; // arg1_ <- learning rate, arg2_ <- momentum, arg3_ <- decay, flag_ <- nesterov
            }
            if constexpr ( ceras::string_equal( static_name, "adagrad" ) || ceras::string_equal( static_name, "Adagrad" ) )
            {
                return ceras::adagrad{ loss, batch_size_, arg1_, arg2_ }; // arg1_ <- learning rate, arg2_ <- decay
            }
            if constexpr ( ceras::string_equal( static_name, "rmsprop" ) || ceras::string_equal( static_name, "RMSprop" ) )
            {
                return ceras::rmsprop{ loss, batch_size_, arg1_, arg2_, arg3_ }; // arg1_ <- learning rate, arg2_ <- rho, arg3_ <- decay
            }
            if constexpr ( ceras::string_equal( static_name, "adam" ) || ceras::string_equal( static_name, "Adam" ) )
            {
                return ceras::adam{ loss, batch_size_, arg1_, arg2_, arg3_, flag_ }; // arg1_ <- learning rate, arg2_ <- beta_1, arg3_ <- beta_2, flag_ <- amsgrad
            }
            if constexpr ( ceras::string_equal( static_name, "adadelta" ) || ceras::string_equal( static_name, "Adadelta" ) )
            {
                return ceras::adadelta{ loss, batch_size_, arg1_ }; // arg1_ <- rho
            }
            // TODO: Adamax, Nadam, Ftrl
            else
            {
                better_assert( false, "Cannot find this optimizer: ", static_name );
            }
        }
    };


}//namespace Keras

#endif//UNNVJXQUIHOGAUQDARAHWUAYFHCTQMPOMMMUELKRAGSVIPSEIQKAUSTGABINSFKPHINELBKVJ

