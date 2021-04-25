#include "../include/ceras.hpp"
#include "../include/layer.hpp"
#include "../include/utils/range.hpp"
#include "../include/utils/better_assert.hpp"
#include "../include/utils/color.hpp"
#include "../include/utils/tqdm.hpp"
#include "../include/utils/imageio.hpp"

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

auto make_encoder()
{
    using namespace ceras;
    auto input = Input();// (28*28, )
    auto l1 = relu( Dense( 256, 28*28 )( input ) );
    auto l2 = relu( Dense( 128, 256 )( l1 ) );
    return model( input, l2 );
}

auto make_decoder( unsigned long const latent_dim )
{
    using namespace ceras;
    auto z = Input(); // (latent_dim, ) -> (28*28,)
    auto z_decoder_1 = relu( Dense( 128, latent_dim )( z ) );
    auto z_decoder_2 = relu( Dense( 256, 128 )( z_decoder_1 ) );
    auto y = sigmoid( Dense( 28*28, 256 )( z_decoder_2 ) );
    return model( z, y );
}

int main()
{
    ceras::random_generator.seed( 42 );
    //load training set
    std::vector<std::uint8_t> training_images = load_binary( training_image_path ); // [u32, u32, u32, u32, uint8, uint8, ... ]

    using namespace ceras;
    typedef tensor<float> tensor_type;
    unsigned long const latent_dim = 2;

    auto encoder = make_encoder();
    auto decoder = make_decoder( latent_dim );

    auto x = Input();
    auto l2 = encoder( x );
    auto z_mean = Dense( latent_dim, 128 )( l2 );
    auto z_log_var = Dense( latent_dim, 128 )( l2 );
    auto z = z_mean + hadamard_product( exponential(value(0.5f)*z_log_var), random_normal_like(0.0f, 1.0f)( z_mean ) );
    auto y = decoder( z );


    auto reconstruction_loss = cross_entropy( x, y );
    auto kl_loss = sum_reduce(  value{-0.5} * (value{1.0} + z_log_var - square(z_mean) - exponential(z_log_var)) );
    auto loss = reconstruction_loss + kl_loss;

    // preparing training
    std::size_t const batch_size = 10;
    tensor_type input_images{ {batch_size, 28*28} };

    //std::size_t const epoch = 100;
    //std::size_t const epoch = 10;
    std::size_t const epoch = 10;
    std::size_t const iteration_per_epoch = 60000/batch_size;

    // creating session
    auto& s = get_default_session<tensor_type>();
    s.bind( x, input_images );

    float learning_rate = 1.0e-3f;
    auto optimizer = gradient_descent{ loss, batch_size, learning_rate };

    for ( [[maybe_unused]] auto e : range( epoch ) )
    {

        for ( [[maybe_unused]] auto i : range( iteration_per_epoch ) )
        {
            // generate images
            std::size_t const image_offset = 16 + i * batch_size * 28 * 28;
            for ( auto j : range( batch_size * 28 * 28 ) )
                input_images[j] = static_cast<float>(training_images[j+image_offset]) / 255.0f;
            better_assert( !has_nan( input_images ), "input_images has nan at iteration ", i );

            auto current_error = s.run( loss );
            std::cout << "Error at epoch " << e << " iteration " << i << ": " << current_error[0] << "\r" << std::flush;
            better_assert( !has_nan(current_error), "Error in current loss." );
            s.run( optimizer );
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;


    unsigned long const n = 30;

    auto grid = linspace( -1.0f, 1.0f, n );
    auto input_data = tensor<float>{ {1, 2,} };
    std::vector<tensor<float>> results;

    for ( auto r : range( n ) )
        for ( auto c : range( n ) )
        {
            input_data[0] = grid[r];
            input_data[1] = grid[c];

            auto result = squeeze( decoder.predict( input_data ) );
            result.reshape( {28, 28} );
            results.push_back( result * 255.0f );
        }

    for ( auto idx : range( results.size() ) )
    {
        std::string const file_name = std::string{"./test/mnist_vae_"} + std::to_string( idx ) + std::string{".png"};
        imageio::imwrite( file_name, results[idx] );
    }

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

