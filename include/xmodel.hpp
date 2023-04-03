#ifndef BPLYFMIFNNWSGMLLEKBJMAJDBRSPHHRAYMOHTWSTCMNMFSLLYNQTTCCAQXKXSLMSLKESHRASL
#define BPLYFMIFNNWSGMLLEKBJMAJDBRSPHHRAYMOHTWSTCMNMFSLLYNQTTCCAQXKXSLMSLKESHRASL

#include "./includes.hpp"
#include "./operation.hpp"
#include "./place_holder.hpp"
#include "./session.hpp"
#include "./tensor.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/context_cast.hpp"
#include "./utils/tqdm.hpp"
#include "./utils/list.hpp"
#include "./utils/debug.hpp"

namespace ceras
{

    template< List Input_List, List Output_List >
    struct model
    {
        Input_List  input_layer_;
        Output_List output_layer_;

        constexpr model( Input_List const& input_layer, Output_List const& output_layer ) : input_layer_{ input_layer }, output_layer_{ output_layer } {}

        ///
        /// @brief Returns the input layer(s) of the model in a 'list', which is are `place_holder`s.
        ///
        auto constexpr input() const { return input_layer_; }

        ///
        /// @brief Returns the output layer(s) of the model in a 'list', which is are expressions.
        ///
        auto constexpr output() const { return output_layer_; }


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
        auto constexpr predict( Tsor const& input_tensor ) const // in case of only one input layer
        {

            //constexpr debug<length( input_layer_ )> = 0;

            better_assert( length( input_layer_ ) == 1, "Expecting the model only have a single input layer, but get ", length( input_layer_ ) );

            auto& s = get_default_session<Tsor>();
            s.bind( car(input_layer_), input_tensor );

            learning_phase = 0; // for different behaviours in normalization and drop-out layers

            if constexpr( length(output_layer_) == 1 )
            {
                // if the model has a single output layer, return a tensor.
                Tsor ans = s.run( car(output_layer_) );
                learning_phase = 1; // restore learning phase
                return ans;
            }
            else
            {
                // if the model has multiple output layer, return a tuple of tensors
                auto ans = map( [&s]<Expression Ex>( Ex const& ex ){ return s.run(ex); }, output_layer_ );
                learning_phase = 1; // restore learning phase
                return ans.as_tuple();
            }
        }

        template< List Tsor_List >
        auto predict( Tsor_List const& input_tensor ) const
        {
            static_assert( length(input_tensor) == length(input_layer_), "Expecting same number of input layers" );

            if constexpr( length(input_tensor) == 1 )
            {
                // case of single input layer
                return predict( car(input_tensor) );
            }
            else
            {
                typedef typename std::remove_cv_t<decltype( car(input_tensor) )> tensor_type;
                auto& s = get_default_session<tensor_type>();
                learning_phase = 0; // supress training behaviours in Dropout and BN layers

                map
                (
                    [&s]<List List_PHolder_Tensor>( List_PHolder_Tensor const& list_of_layer_and_tensor ) // invokes on sub-list of input_layer-input-tensor
                    {
                        list_of_layer_and_tensor
                        (
                            [&s]<Place_Holder Ph, Tensor Tsor>(Ph input_layer, Tsor input_tensor) // invokes with an input layer (a place holder) and an input tensor
                            {
                                s.bind( input_layer, input_tensor );
                            }
                        );
                    },
                    zip( input_layer_, input_tensor ) // pairing input layer and input tensor as a list of list -> [ [input_layer_0, input_tensor_0], [input_layer_1, input_tensor_1], ..., [input_layer_n, input_tensor_n] ]
                );

                if constexpr( length(output_layer_) == 1 )
                {
                    // if the model has a single output layer, return a tensor.
                    tensor_type ans = s.run( car(output_layer_) );
                    learning_phase = 1; // restore learning phase
                    return ans;
                }
                else
                {
                    // if the model has multiple output layer, return a tuple of tensors
                    auto ans = map( [&s]<Expression Ex>( Ex const& ex ){ return s.run(ex); }, output_layer_ );
                    learning_phase = 1; // restore learning phase
                    return ans.as_tuple();
                }

            }
        }


    }; // struct model


}//namespace ceras

#endif//BPLYFMIFNNWSGMLLEKBJMAJDBRSPHHRAYMOHTWSTCMNMFSLLYNQTTCCAQXKXSLMSLKESHRASL
