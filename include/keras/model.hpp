#ifndef PUQHHDUYRUDLVEAWDMAXQUVLDEJPSILRTQWNXXAUEFUKPVFNOVJFOOBWBYKTEVYVPXPKOEROX
#define PUQHHDUYRUDLVEAWDMAXQUVLDEJPSILRTQWNXXAUEFUKPVFNOVJFOOBWBYKTEVYVPXPKOEROX

#include "./layer.hpp"
#include "./optimizer.hpp"
#include "./loss.hpp"
#include "../tensor.hpp"
#include "../utils/debug.hpp"
#include "../utils/range.hpp"
#include "../utils/tqdm.hpp"

namespace Keras
{

    namespace keras_details
    {
        template< typename Optimizer, typename Loss, typename Model >
        auto create_model_context( Model model )
        {
            ceras::debug_print("creating model context.");
            auto input = model.input_;
            auto output = model.output_;
            auto loss = Loss{};
            auto optimizer = Optimizer{};
            return make_tuple( input, output, loss, optimizer );
        }

        template< typename Model_Context >
        struct compiled_model
        {
            Model_Context model_context_;

            compiled_model( Model_Context const& model_context ) noexcept : model_context_{ model_context } {}

            template< ceras::Tensor Tsor >
            Tsor predict( Tsor const& x ) // should have a batch size
            {
                auto [input, output, loss, optimizer] = model_context_;
                ceras::session<Tsor> s;
                auto c_input = input();
                auto c_output = output();
                s.bind( c_input, x );
                return s.run( c_output );
            }

            template< ceras::Tensor Tsor >
            std::vector<float> fit( Tsor const& x, Tsor const& y, unsigned long batch_size=32, unsigned long epochs=10 )//, bool verbose=true, bool shuffle=true )
            {
                auto [input, output, loss, optimizer] = model_context_;
                ceras::session<Tsor> s;
                auto c_input = input();
                auto c_output = output();
                auto c_gt = ceras::place_holder<ceras::tensor<float>>{};
                auto c_loss = loss(c_gt, c_output);
                auto c_optimizer = optimizer( c_loss );

                //preparing dataset
                unsigned long const length_x = *(x.shape().begin());
                unsigned long const length_y = *(y.shape().begin());
                better_assert( length_x == length_y, "first dim of x and y should be equal, but length_x = ", length_x, " and length_y = ", length_y );

                unsigned long const loops_per_epoch = length_x / batch_size;

                std::vector<float> errors;

                for ( [[maybe_unused]]auto e : ceras::range( epochs ) )
                {
                    for ( auto l : tq::dm(tq::range(loops_per_epoch)) )
                    {
                        auto xx = x.slice(l*batch_size, (l+1)*batch_size);
                        auto yy = y.slice(l*batch_size, (l+1)*batch_size);
                        s.bind( c_input, x );
                        s.bind( c_gt, y );
                        auto error = s.run( c_loss );
                        s.run( c_optimizer );
                        errors.push_back( error[0] );
                    }
                }
                return errors;
            }
        };

        //
        //TODO: for multi-input-and-output cases, define intemediate models
        //

        template< typename Input, typename Output >
        struct model
        {
            Input input_;
            Output output_;

            model( Input const& input, Output const& output ) noexcept : input_{input}, output_{output} {}

            template< typename Optimizer, typename Loss > // metrics, loss_weights, weighted_metrics,
            auto compile() const noexcept
            {
                return compiled_model{ create_model_context<Optimizer, Loss, model<Input, Output>>( *this ) };
            }
        };
    }

    //template< typename ... Args >
    //using Model = keras_details::model<Args...>;
    // 'cause of the gcc bug 99118: <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=99118>
    template< typename Input, typename Output >
    struct Model : keras_details::model<Input, Output>
    {
        Model( Input const& inputs, Output const& outputs ) noexcept : keras_details::model<Input, Output>{ inputs, outputs } {}
    };

}//namespace Keras

#endif//PUQHHDUYRUDLVEAWDMAXQUVLDEJPSILRTQWNXXAUEFUKPVFNOVJFOOBWBYKTEVYVPXPKOEROX

