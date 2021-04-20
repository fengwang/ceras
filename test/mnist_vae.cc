#include "../include/ceras.hpp"
#include "../include/layer.hpp"
#include "../include/utils/range.hpp"
#include "../include/utils/better_assert.hpp"
#include "../include/utils/color.hpp"
#include "../include/utils/tqdm.hpp"

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
    ceras::random_generator.seed( 42 );
    //load training set
    std::vector<std::uint8_t> training_images = load_binary( training_image_path ); // [u32, u32, u32, u32, uint8, uint8, ... ]

    using namespace ceras;
    typedef tensor<float> tensor_type;
    unsigned long latent_dim = 2;
    auto x = place_holder<tensor_type>{}; // 1-D, 28x28 pixels
    auto l1 = relu( Dense( 256, 28*28 )( x ) );
    auto l2 = relu( Dense( 128, 256 )( l1 ) );
    auto z_mean = Dense( latent_dim, 256 )( l2 );
    auto z_log_var = Dense( latent_dim, 256 )( l2 );
    //auto z = z_mean + hadamard_product( exp(z_log_var), random_normal_like(0.0f, 1.0f)( z_mean ) );
    //auto z = z_mean;// + hadamard_product( exp(z_log_var), random_normal_like(0.0f, 1.0f)( z_mean ) );
    //auto z = elementwise_exp( z_log_var );
    auto z = exp( z_log_var );

    auto z_decoder_1 = relu( Dense( 128, latent_dim )( z ) );
    auto z_decoder_2 = relu( Dense( 256, 128 )( z_decoder_1 ) );
    auto y = sigmoid( Dense( 28*28, 256 )( z_decoder_2 ) );

    auto reconstruction_loss = sum_reduce( cross_entropy( x, y ) );
    auto kl_loss = sum_reduce( value{-0.5} * (value{1.0} + z_log_var - square(z_mean) - exp(z_log_var)) ) ;
    //auto loss = reconstruction_loss + kl_loss;
    auto loss = reconstruction_loss;

    // preparing training
    std::size_t const batch_size = 10;
    tensor_type input_images{ {batch_size, 28*28} };
    tensor_type output_labels{ {batch_size, 10} };

    //std::size_t const epoch = 100;
    std::size_t const epoch = 10;
    std::size_t const iteration_per_epoch = 60000/batch_size;

    // creating session
    session<tensor_type> s;
    s.bind( x, input_images );

    float learning_rate = 1.0e-1f;
    auto optimizer = gradient_descent{ loss, batch_size, learning_rate };

#if 1

    for ( [[maybe_unused]] auto e : range( epoch ) )
    {

        for ( [[maybe_unused]] auto i : tq::trange( iteration_per_epoch ) )
        {
            // generate images
            std::size_t const image_offset = 16 + i * batch_size * 28 * 28;
            for ( auto j : range( batch_size * 28 * 28 ) )
                input_images[j] = static_cast<float>(training_images[j+image_offset]) / 127.5f - 1.0f;
            better_assert( !has_nan( input_images ), "input_images has nan at iteration ", i );

            auto current_error = s.run( loss );
            better_assert( !has_nan(current_error), "Error in current loss." );
            s.run( optimizer );
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
#endif

#if 0
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

