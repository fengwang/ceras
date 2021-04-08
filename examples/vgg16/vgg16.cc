#include "../../include/ceras.hpp"
#include "../../include/utils/range.hpp"
#include "../../include/utils/better_assert.hpp"

#include <fstream>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <iterator>


using namespace ceras;
typedef tensor<float> tensor_type;

//
// example: Relu_Conv( 63, 3, {224, 224, 3} )( input_of_shape_224x224x3 );
//
inline auto Relu_Conv2D( unsigned long output_channels,std::vector<unsigned long> const& kernel_size, std::vector<unsigned long> const& input_shape )
{
    return [=]<Expression Ex>( Ex const& ex )
    {
        unsigned long const kernel_size_x = kernel_size[0];
        unsigned long const kernel_size_y = kernel_size[1];
        unsigned long const input_channels = input_shape[2];
        unsigned long const input_x = input_shape[0];
        unsigned long const input_y = input_shape[1];
        auto w = variable<tensor_type>{ glorot_uniform<float>({output_channels, kernel_size_x, kernel_size_y, input_channels}) };
        auto b = variable<tensor_type>{ zeros<float>( {1, 1, output_channels} ) };
        return relu( conv2d( input_x, input_y, 1, 1, 1, 1, "same" )( ex, w ) + b );
    };
}

//
// example: Relu_Dense( 512, 17 )( input_of_shape_17 );
//
inline auto Relu_Dense( unsigned long output_size, unsigned long input_size )
{
    return [=]<Expression Ex>( Ex const& ex )
    {
        auto w = variable<tensor_type>{ glorot_uniform<float>({input_size, output_size}) };
        auto b = variable<tensor_type>{ zeros<float>({1, output_size}) };
        return relu( ex * w + b );
    };
}

int main()
{
    random_generator.seed( 42 ); // just for reproducibility

    auto input = place_holder<tensor_type>{}; //  3D tensor input, (batch_size, 224, 224, 3)
    auto l0 = Relu_Conv2D( 64, {3, 3}, {224, 224, 3} )( input ); // 224, 224, 64
    auto l1 = max_pooling_2d( 2 ) ( l0 ); // 112, 112, 64
    auto l2 = Relu_Conv2D( 128, {3, 3}, {112, 112, 64} )( l1 ); // 112, 112, 128
    auto l3 = Relu_Conv2D( 128, {3, 3}, {112, 112, 128} )( l2 ); // 112, 112, 128
    auto l4 = max_pooling_2d( 2 ) ( l3 ); // 56, 56, 128
    auto l5 = Relu_Conv2D( 256, {3, 3}, {56, 56, 128} )( l4 ); // 56, 56, 256
    auto l6 = Relu_Conv2D( 256, {3, 3}, {56, 56, 256} )( l5 ); // 56, 56, 256
    auto l7 = Relu_Conv2D( 256, {3, 3}, {56, 56, 256} )( l6 ); // 56, 56, 256
    auto l8 = max_pooling_2d( 2 ) ( l7 ); // 28, 28, 256
    auto l9 = Relu_Conv2D( 512, {3, 3}, {28, 28, 256} )( l8 ); // 28, 28, 512
    auto l10 = Relu_Conv2D( 512, {3, 3}, {28, 28, 512} )( l9 ); // 28, 28, 512
    auto l11 = Relu_Conv2D( 512, {3, 3}, {28, 28, 512} )( l10 ); // 28, 28, 512
    auto l12 = max_pooling_2d( 2 ) ( l11 ); // 14, 14, 512
    auto l13 = Relu_Conv2D( 512, {3, 3}, {14, 14, 512} )( l12 ); // 14, 14, 512
    auto l14 = Relu_Conv2D( 512, {3, 3}, {14, 14, 512} )( l13 ); // 14, 14, 512
    auto l15 = Relu_Conv2D( 512, {3, 3}, {14, 14, 512} )( l14 ); // 14, 14, 512
    auto l16 = max_pooling_2d( 2 ) ( l15 ); // 7, 7, 512
    auto l17 = flatten( l16 ); // 7x7x512
    auto l18 = Relu_Dense( 4096, 7*7*512 )( l17 ); // 4096
    auto l19 = Relu_Dense( 4096, 4096 )( l18 ); // 4096
    auto l20 = Relu_Dense( 1000, 4096 )( l19 ); // 1000
    auto output = l20;

    auto ground_truth = place_holder<tensor_type>{}; // 1-D, 1000
    auto loss = cross_entropy_loss( ground_truth, output );




    #if 0
    // preparing training
    std::size_t const batch_size = 10;
    tensor_type input_images{ {batch_size, 224, 224, 3} };
    tensor_type output_labels{ {batch_size, 1000} };

    //std::size_t const epoch = 100;
    std::size_t const epoch = 1;
    std::size_t const iteration_per_epoch = 60000/batch_size;

    // creating session
    session<tensor_type> s;
    s.bind( input, input_images );
    s.bind( ground_truth, output_labels );

    // proceed training
    float learning_rate = 1.0e-1f;
    //float learning_rate = 1.0e-1f;
    //float learning_rate = 5.0e-1f;
    auto optimizer = gradient_descent{ loss, batch_size, learning_rate };

    for ( auto e : range( epoch ) )
    {

        for ( auto i : range( iteration_per_epoch ) )
        {
            // generate images
            std::size_t const image_offset = 16 + i * batch_size * 28 * 28;
            for ( auto j : range( batch_size * 28 * 28 ) )
                input_images[j] = static_cast<float>(training_images[j+image_offset]) / 127.5f - 1.0f;
            better_assert( !has_nan( input_images ), "input_images has nan at iteration ", i );

            // generating labels
            std::size_t const label_offset = 8 + i * batch_size * 1;
            std::fill_n( output_labels.data(), output_labels.size(), 0.0f ); //reset
            for ( auto j : range( batch_size * 1 ) )
            {
                std::size_t const label = static_cast<std::size_t>(training_labels[j+label_offset]);
                output_labels[j*10+label] = 1.0f;
            }
            better_assert( !has_nan( output_labels ), "output_labels has nan at iteration ", i );

            auto current_error = s.run( loss );
            std::cout << "Loss at epoch " << e << " index: " << (i+1)*batch_size << ":\t" << current_error[0] << "\r" << std::flush;
            better_assert( !has_nan(current_error), "Error in current loss." );
            s.run( optimizer );
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;


    unsigned long const new_batch_size = 1;

    std::vector<std::uint8_t> testing_images = load_binary( testing_image_path );
    std::vector<std::uint8_t> testing_labels = load_binary( testing_label_path );
    std::size_t const testing_iterations = 10000 / new_batch_size;

    tensor<float> new_input_images{ {new_batch_size, 28 * 28} };
    s.bind( input, new_input_images );

    unsigned long errors = 0;

    for ( auto i = 0UL; i != testing_iterations; ++i )
    {
        std::size_t const image_offset = 16 + i * new_batch_size * 28 * 28;

        for ( auto j = 0UL; j != new_batch_size*28*28; ++j )
            new_input_images[j] = static_cast<float>( testing_images[j + image_offset] ) / 127.5f - 1.0f;

        auto prediction = s.run( output );
        prediction.reshape( {prediction.size(), } );
        std::size_t const predicted_number = std::max_element( prediction.begin(), prediction.end() ) - prediction.begin();

        std::size_t const label_offset = 8 + i * new_batch_size * 1;
        std::size_t const ground_truth = testing_labels[label_offset];

        if ( predicted_number != ground_truth )
        {
            errors += 1;
            std::cout << "Prediction error at " << i << ": predicted " << predicted_number << ", but the ground_truth is " << ground_truth << std::endl;
        }

    }

    float const err = 1.0 * errors / 10000;
    std::cout << "Prediction error on the testing set is " << err << std::endl;

    #endif

    return 0;
}

