#include "../include/ceras.hpp"
#include "../include/utils/range.hpp"
#include "../include/utils/better_assert.hpp"
#include "../include/utils/color.hpp"

#include <fstream>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <iterator>

// image: [u32, u32, u32, u32, uint8, uint8 .... ]
// label: [u32, u32, uint8, uint8 .... ]
std::string const training_image_path{ "./examples/mnist/dataset/mnist/train-images-idx3-ubyte" }; // 60000 images
std::string const training_label_path{ "./examples/mnist/dataset/mnist/train-labels-idx1-ubyte" }; // 60000 labels
std::string const testing_image_path{ "./examples/mnist/dataset/mnist/t10k-images-idx3-ubyte" }; // 10000 images
std::string const testing_label_path{ "./examples/mnist/dataset/mnist/t10k-labels-idx1-ubyte" }; // 10000 labels

std::vector<std::uint8_t> load_binary( std::string const& filename )
{
    std::ifstream ifs( filename, std::ios::binary );
    better_assert( ifs.good(), "Failed to load data from ", filename );
    std::vector<char> buff{ ( std::istreambuf_iterator<char>( ifs ) ), ( std::istreambuf_iterator<char>() ) };
    std::vector<std::uint8_t> ans( buff.size() );
    std::copy( buff.begin(), buff.end(), reinterpret_cast<char*>( ans.data() ) );
    std::cout << "Loaded binary from file " << color::rize( filename, "Red" ) << ", got " << color::rize( buff.size(), "Green" ) << " bytes." << std::endl;
    return ans;
}

int main()
{
    //load training set
    std::vector<std::uint8_t> training_images = load_binary( training_image_path ); // [u32, u32, u32, u32, uint8, uint8, ... ]
    std::vector<std::uint8_t> training_labels = load_binary( training_label_path ); // [u32, u32, uint8, uint8, ... ]


    // define computation graph, a 3-layered dense net with topology 784x256x128x10
    using namespace ceras;
    typedef tensor<float> tensor_type;
    auto input = place_holder<tensor_type>{}; // 1-D, 28x28 pixels

    auto l0 = reshape( {28, 28, 1} )( input );

    auto k1 = variable{ randn<float>( {32, 3, 3, 1}, 0.0, 10.0/std::sqrt(32.0*3*3*1) ) };
    auto l1 = relu( conv2d(28, 28, 1, 1, 1, 1, "valid" )( l0, k1 ) ); // 26, 26, 32

    auto l2 = max_pooling_2d( 2 ) ( l1 ); // 13, 13, 32

    auto k2 = variable{ randn<float>( {64, 3, 3, 32}, 0.0, 10.0/std::sqrt(64.0*3*3*1) ) };
    auto l3 = relu( conv2d(13, 13, 1, 1, 1, 1, "valid")( l2, k2 ) ); // 11, 11, 64

    auto l4 = max_pooling_2d( 2 )( l3 ); //5, 5, 64
    auto l5 = drop_out(0.5)( flatten( l4 ) );

    auto w6 = variable{ randn<float>( {5*5*64, 10}, 0.0, 10.0/std::sqrt(7.0*7*64*10) ) };
    auto b6 = variable{ zeros<float>( {1, 10} ) };

    auto l6 = l5 * w6 + b6;
    auto output = l6;

    auto ground_truth = place_holder<tensor_type>{}; // 1-D, 10
    auto loss = cross_entropy_loss( ground_truth, output );

    // preparing training
    std::size_t const batch_size = 5;
    tensor<float> input_images{ {batch_size, 28*28} };
    tensor<float> output_labels{ {batch_size, 10} };

    std::size_t const epoch = 2;
    std::size_t const iteration_per_epoch = 60000/batch_size;

    // creating session
    auto& s = get_default_session<tensor_type>();
    //auto& s = get_default_session<tensor_type>().get();
    s.bind( input, input_images );
    s.bind( ground_truth, output_labels );

    // proceed training
    float learning_rate = 1.0e-3f;
    auto optimizer = gradient_descent{ loss, batch_size, learning_rate };

    ceras::learning_phase = 1;

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

    ceras::learning_phase = 0;

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

    return 0;
}

