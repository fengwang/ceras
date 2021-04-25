#ifndef BPLYFMIFNNWSGMLLEKBJMAJDBRSPHHRAYMOHTWSTCMNMFSLLYNQTTCCAQXKXSLMSLKESHRASL
#define BPLYFMIFNNWSGMLLEKBJMAJDBRSPHHRAYMOHTWSTCMNMFSLLYNQTTCCAQXKXSLMSLKESHRASL

#include "./includes.hpp"
#include "./operation.hpp"
#include "./place_holder.hpp"
#include "./tensor.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/context_cast.hpp"
#include "./utils/tqdm.hpp"

namespace ceras
{

    template< Expression Ex >
    void make_trainable( Ex& ex , bool t )
    {
        if constexpr (is_variable_v<Ex>)
        {
            ex.trainable( t );
        }
        else if constexpr (is_binary_operator_v<Ex>)
        {
            make_trainable( ex.lhs_op_, t );
            make_trainable( ex.rhs_op_, t );
        }
        else if constexpr (is_unary_operator_v<Ex>)
        {
            make_trainable( ex.op_, t );
        }
    }

    ///
    /// Replacing a place_holder with an expression.
    ///
    /// @param ex Can be a unary operator, binary operator, variable, place_holder, a constant or a value
    /// @param old_place_holder An place holder in @p ex
    /// @param new_expression An expression that will replace @p old_place_holder in @p ex.
    /// @return A expression inheriting the topology of @p ex, but with @p old_place_holder replaced by @p new_expression
    ///
    template< Expression Ex, Place_Holder Ph, Expression Ey >
    auto replace_placeholder_with_expression( Ex const& ex, Ph const& old_place_holder, Ey const& new_expression )
    {
        if constexpr (is_value_v<Ex> || is_constant_v<Ex> || is_variable_v<Ex>)
        {
            return ex;
        }
        else if constexpr (is_place_holder_v<Ex>)
        {
            return new_expression; // assuming only one place holder in the model
            //return (ex == old_place_holder) ? new_expression : ex;
        }
        else if constexpr (is_unary_operator_v<Ex>)
        {
            return make_unary_operator( ex.forward_action_, ex.backward_action_, ex.name_, ex.reset_action_ )( replace_placeholder_with_expression( ex.op_, old_place_holder, new_expression ) );
        }
        else if constexpr (is_binary_operator_v<Ex>)
        {
            return make_binary_operator( ex.forward_action_, ex.backward_action_, ex.name_, ex.reset_action_ )
                                       ( replace_placeholder_with_expression( ex.lhs_op_, old_place_holder, new_expression ),
                                         replace_placeholder_with_expression( ex.rhs_op_, old_place_holder, new_expression ) );
        }
        else
        {
            better_assert( false, "replace::Should never reach here!" );
        }
    }

    template< typename Model, typename Optimizer, typename Loss >
    struct compiled_model
    {
        typedef typename Model::input_layer_type io_layer_type;
        typedef decltype(std::declval<Optimizer>()(std::declval<Loss&>())) optimizer_type; // defined because the compiled optimizer takes a reference to an expression as its parameter

        Model model_;
        io_layer_type input_place_holder_;
        io_layer_type ground_truth_place_holder_;
        Loss loss_; // MeanSquaredError()( model_.output() )( find a input );
        Optimizer optimizer_; // Adam( ... )
        optimizer_type compiled_optimizer_;

        compiled_model( Model const& m, io_layer_type const& input_place_holder, io_layer_type const& ground_truth_place_holder, Loss const& loss, Optimizer const& optimizer ):
            model_{m}, input_place_holder_{input_place_holder}, ground_truth_place_holder_{ground_truth_place_holder}, loss_{loss}, optimizer_{optimizer}, compiled_optimizer_{ optimizer_(loss_) }  { }

        ///
        /// Calculate the loss for the model in test model.
        /// @param inputs Input data. A tensor of shape (samples, input_shape).
        /// @param outputs Output data. A tensor of shape (samples, output_shape).
        /// @param batch_size Number of samples per batch of computation. Default to 32.
        /// @return Test loss. A scalar.
        ///
        template< Tensor Tsor >
        auto evaluate( Tsor const& inputs, Tsor const& outputs, unsigned long batch_size=32 )
        {
            // extract size of samples
            unsigned long const samples = *(inputs.shape().begin());
            unsigned long const loops = samples / batch_size;

            // prepare tensor for inputs
            std::vector<unsigned long> batch_input_shape = inputs.shape();
            batch_input_shape[0] = batch_size;
            Tsor input_samples{ batch_input_shape };
            unsigned long const input_size_per_batch = input_samples.size();

            // prepare tensor for outputs
            std::vector<unsigned long> batch_output_shape = outputs.shape();
            batch_output_shape[0] = batch_size;
            Tsor output_samples{ batch_output_shape };
            unsigned long const output_size_per_batch = output_samples.size();

            // bind tensors to place holders
            //session<Tsor> s;
            auto& s = get_default_session<Tsor>();//.get();
            s.bind( input_place_holder_, input_samples );
            s.bind( ground_truth_place_holder_, output_samples );

            typedef typename Tsor::value_type value_type;
            value_type validation_error = 0;

            learning_phase = 0; // for different behaviours in normalization and drop-out layers

            for ( auto l : tq::trange( loops ) )
            {
                // feed data
                std::copy_n( inputs.data() + l * input_size_per_batch, input_size_per_batch, input_samples.data() );
                std::copy_n( outputs.data() + l * output_size_per_batch, output_size_per_batch, output_samples.data() );
                // forward pass
                auto error = s.run( loss_ ).as_scalar();
                // in case of training split, do backpropagation
                validation_error += error;
            }

            learning_phase = 1; // for different behaviours in normalization and drop-out layers

            return validation_error / loops;
        }

        ///
        /// Train the model on the selected dataset for a fixed numbers of epoches.
        /// @param inputs Input data. A tensor of shape (samples, input_shape).
        /// @param outputs Input data. A tensor of shape (samples, output_shape).
        /// @param batch_size Number of samples per gradient update. Should agree with the batch size in the optimizer.
        /// @param epoch Number of epoches to train the dataset.
        /// @param verbose Verbosity mode. 0 for slient. 1 for one line per epoch.
        /// @param validation_split Fraction of the training data that will be used for validation. A floating number in range [0, 1].
        /// @return A tuple of two vectors. The first vector gives the historical errors on the training data. The second vector gives the historical errors on the validation data.
        ///
        /// Example:
        /// @code
        /// model m{ ... };
        /// auto cm = m.compile( ... );
        /// tensor<float> inputs, outputs;
        /// //...
        /// unsigned long batch_size = 32;
        /// unsigned long epoch = 10;
        /// int verbose = 1;
        /// double validation_split = 0.2;
        /// auto errors = cm.fit( inputs, outputs, batch_size, epoch, verbose, validation_split );
        /// @endcode
        ///
        template< Tensor Tsor >
        auto fit( Tsor const& inputs, Tsor const& outputs, unsigned long batch_size, unsigned long epoch=1, int verbose=0, double validation_split=0.0 )
        {
            // extract size of samples
            unsigned long const samples = *(inputs.shape().begin());
            unsigned long const loops_per_epoch = samples / batch_size;
            unsigned long const training_loops = ( 1.0 - validation_split ) * loops_per_epoch;
            unsigned long const validation_loops = loops_per_epoch - training_loops;

            // prepare tensor for inputs
            std::vector<unsigned long> batch_input_shape = inputs.shape();
            batch_input_shape[0] = batch_size;
            Tsor input_samples{ batch_input_shape };
            unsigned long const input_size_per_batch = input_samples.size();

            // prepare tensor for outputs
            std::vector<unsigned long> batch_output_shape = outputs.shape();
            batch_output_shape[0] = batch_size;
            Tsor output_samples{ batch_output_shape };
            unsigned long const output_size_per_batch = output_samples.size();

            // bind tensors to place holders
            //session<Tsor> s;
            auto& s = get_default_session<Tsor>();//.get();
            s.bind( input_place_holder_, input_samples );
            s.bind( ground_truth_place_holder_, output_samples );

            // collect training errors
            typedef typename Tsor::value_type value_type;
            std::vector<value_type> training_errors;
            std::vector<value_type> validation_errors;

            learning_phase = 1; // for different behaviours in normalization and drop-out layers

            for ( auto e : range( epoch ) )
            {
                value_type training_error = 0;
                value_type validation_error = 0;
                for ( auto l : tq::trange( loops_per_epoch ) )
                {
                    // feed data
                    std::copy_n( inputs.data() + l * input_size_per_batch, input_size_per_batch, input_samples.data() );
                    std::copy_n( outputs.data() + l * output_size_per_batch, output_size_per_batch, output_samples.data() );
                    // forward pass
                    auto error = s.run( loss_ ).as_scalar();
                    // in case of training split, do backpropagation
                    if ( l <= training_loops )
                    {
                        training_error += error;
                        s.run( compiled_optimizer_ );
                    }
                    else // in case of validation split, just collect errors
                    {
                        validation_error += error;
                    }
                }
                training_errors.push_back( training_error / training_loops );
                validation_errors.push_back( validation_error / validation_loops );
                if ( verbose )
                    std::cout << "\nTraining error: " << training_error / training_loops << " and validation error: " << validation_error / validation_loops << " at epoch: " << e+1 << "/" << epoch;
                std::cout << std::endl;
            }
            return std::make_tuple( training_errors, validation_errors );
        }

        ///
        /// Running a single updated on a single batch of data.
        /// @param input The input data to train the model. A tensor of shape (batch_size, input_shape).
        /// @param output The output data to train the model. A tensor of shape (batch_size, output_shape).
        /// @return Training loss. A scalar.
        ///
        /// Example code:
        /// @code
        /// auto m = model{ ... };
        /// auto cm = m.compile( ... );
        /// for ( auto idx : range( 1024 ) )
        /// {
        ///     auto x = ...; // get batch input
        ///     auto y = ...; // get batch output
        ///     cm.train_on_batch( x, y );
        /// }
        /// @endcode
        ///
        template< Tensor Tsor >
        auto train_on_batch( Tsor const& input, Tsor const& output )
        {
            learning_phase = 1; // for different behaviours in normalization and drop-out layers
            //session<Tsor> s;
            auto& s = get_default_session<Tsor>();//.get();
            s.bind( input_place_holder_, input );
            s.bind( ground_truth_place_holder_, output );
            auto error = s.run( loss_ );
            s.run( compiled_optimizer_ );
            return error.as_scalar();
        }

        template< Tensor Tsor>
        auto predict( Tsor const& input_tensor )
        {
            auto m = model_;
            return m.predict( input_tensor );
        }

        template< Expression Exp >
        auto operator()( Exp const& ex ) const noexcept
        {
            return model_( ex );
        }

        void trainable( bool t )
        {
            return model_.trainable( t );
        }
    };

    template< typename Model, typename Optimizer, typename Loss >
    inline auto make_compiled_model( Model const& m, Loss const& l, Optimizer const& o )
    {
        auto input_place_holder = m.input();
        auto ground_truth_place_holder = typename Model::input_layer_type{};
        auto loss = l( m.output() )( ground_truth_place_holder );
        auto optimizer = o( loss );
        return compiled_model{ m, input_place_holder, ground_truth_place_holder, loss, o};
    }

    ///
    /// Groups an input layer (a place holder) and an output layer (an expression template) into an object.
    ///
    /// @tparam Ex The expression template for the output layer.
    /// @tparam Ph The place holder expression for the input layer
    ///
    template< Expression Ex, Place_Holder Ph >
    struct model
    {
        typedef Ph       input_layer_type;
        typedef Ex       output_layer_type;

        output_layer_type expression_;   ///< output layer of the model.
        input_layer_type place_holder_;//< input layer of the model.


        ///
        /// Returns the input layer of the model, which is a place_holder.
        ///
        input_layer_type input() const noexcept { return place_holder_; }

        ///
        /// Returns the output layer of the model.
        ///
        output_layer_type output() const noexcept { return expression_; }

        ///
        /// @param place_holder The input layer of the model, a place holder.
        /// @param expression The output layer of the model, a expression template.
        ///
        /// Example code to generate a model:
        /// @code
        /// auto input = Input();
        /// auto l1 = relu( Dense( 1024, 28*28 )( input ) );
        /// auto output = sigmoid( Dense( 10, 1024 )( l1 ) );
        /// auto m = model{ input, output };
        /// @endcode
        ///
        model( input_layer_type const& place_holder, output_layer_type const& expression ) :  expression_{expression}, place_holder_{place_holder} {}

        ///
        /// Making prediction by binding the nput data to the `place_holder_` and evaluating `expression_`.
        /// @param input_tensor The input samples.
        /// @return The result this model predicts.
        ///
        /// Example to predict
        /// @code
        /// auto input = Input();
        /// auto l1 = relu( Dense( 1024, 28*28 )( input ) );
        /// auto output = sigmoid( Dense( 10, 1024 )( l1 ) );
        /// // ... train the model after defining a loss and an optimizer
        /// auto m = model{ input, output };
        /// auto test_data = random( {128, 28*28} ); // batch size is 128
        /// auto result = model.predict( test_data ); // should produce an tensor of (128, 10)
        /// @endcode
        ///
        template< Tensor Tsor>
        auto predict( Tsor const& input_tensor )
        {
            learning_phase = 0; // for different behaviours in normalization and drop-out layers

            //session<Tsor> s;
            auto& s = get_default_session<Tsor>();//.get();
            s.bind( place_holder_, input_tensor );

            auto ans = s.run( expression_ );

            learning_phase = 0; // restore learning phase

            return ans;
        }

        ///
        /// Generating a new expression by using the current model.
        /// @param ex An expression that represents the input to the model.
        /// @return An expression that replacing the input node with a new epxression.
        ///
        /// Example code
        /// @code
        /// auto x = Input(); // input, (28*28,)
        /// auto y = Dense( 128, 28*28 )( x );
        /// auto m1 = model( x, y ); // this model is [(28*28,) -> (128,)]
        ///
        /// auto u = Input(); // new input, (32,)
        /// auto v = Dense( 28*28, 32 )( u );
        /// auto m2 = model( u, v );
        ///
        /// auto input = Input(); // (32, )
        /// auto ouptut = m1( m2( input ) ); // this new expression is [(32,) -> (28*28,) -> (128,)], note x is not in this expression any more
        /// auto m = model( input, output ); // create a new model
        /// @endcode
        ///
        template< Expression Exp >
        auto operator()( Exp const& ex ) const noexcept
        {
            return replace_placeholder_with_expression( expression_, place_holder_, ex );
        }

        ///
        /// Compile the model for training
        /// @param l The loss to minimize.
        /// @param o The optimizer to do the optimization.
        /// @return An instance of compiled_model.
        ///
        /// Example useage:
        /// @code
        /// model m{ ... };
        /// unsigned long batch_size = 16;
        /// float learning_rate = 0.001f;
        /// auto cm = m.compile( MeanSquaredError(), SGD( batch_size, learning_rate ) );
        /// @endcode
        ///
        template< typename Loss, typename Optimizer >
        auto compile( Loss const& l, Optimizer const& o )
        {
            return make_compiled_model( *this, l, o );
        }

        ///
        ///
        void trainable( bool t )
        {
            make_trainable( expression_, t );
        }
    };

}//namespace ceras

#endif//BPLYFMIFNNWSGMLLEKBJMAJDBRSPHHRAYMOHTWSTCMNMFSLLYNQTTCCAQXKXSLMSLKESHRASLCalculate the loss for the model in test model
