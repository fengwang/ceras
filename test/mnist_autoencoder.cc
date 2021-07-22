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

auto build_autoencoder(unsigned long threshold)
{
    using namespace ceras;
    auto input = Input();
    auto l1 = relu( Dense( 512, 28*28 )( input ) );
    auto l2 = relu( Dense( 256, 512 )( l1 ) );
    auto l3 = relu( Dense( threshold, 256 )( l2 ) );
    auto l4 = relu( Dense( 256, threshold )( l3 ) );
    auto l5 = relu( Dense( 512, 256 )( l4 ) );
    auto output = sigmoid( Dense( 28*28, 512 )( l5 ) );
    return model{ input, output };
}

int main()
{
    using namespace ceras;
    typedef tensor<float> tensor_type;
    random_generator.seed( 42 );
    //load training set
    std::vector<std::uint8_t> training_images = load_binary( training_image_path ); // [u32, u32, u32, u32, uint8, uint8, ... ]
    std::vector<std::uint8_t> training_labels = load_binary( training_label_path ); // [u32, u32, uint8, uint8, ... ]

    unsigned long threshold = 8;
    auto input = Input(); // (28*28, )
    auto autoencoder = build_autoencoder(threshold);
    auto output = autoencoder( input );
    auto ground_truth = place_holder<tensor_type>{}; // ( 28*28, )
    auto loss = mse( ground_truth, output );

    // preparing training
    std::size_t const batch_size = 10;
    tensor_type input_images{ {batch_size, 28*28} };

    std::size_t const epoch = 10;
    std::size_t const iteration_per_epoch = 60000/batch_size;

    // creating session
    auto& s = get_default_session<tensor_type>();
    s.bind( input, input_images );
    s.bind( ground_truth, input_images );

    // proceed training
    float learning_rate = 5.0e-3f;
    auto optimizer = gradient_descent{ loss, batch_size, learning_rate };

    for ( [[maybe_unused]] auto e : range( epoch ) )
    {

        for ( [[maybe_unused]] auto i : range( iteration_per_epoch ) )
        {
            // generate images
            std::size_t const image_offset = 16 + i * batch_size * 28 * 28;
            for ( auto j : range( batch_size * 28 * 28 ) )
                input_images[j] = static_cast<float>(training_images[j+image_offset]) / 255.0f;
                //input_images[j] = static_cast<float>(training_images[j+image_offset]) / 127.5f - 1.0f;
            better_assert( !has_nan( input_images ), "input_images has nan at iteration ", i );

            auto current_error = s.run( loss );
            std::cout << "Loss at epoch " << e << " index: " << (i+1)*batch_size << ":\t" << current_error[0] << "\r" << std::flush;
            better_assert( !has_nan(current_error), "Error in current loss." );
            s.run( optimizer );
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    // Test
    unsigned long const loops = 32;
    tensor_type prediction = random<float>( {1, 28*28 } );
    for ( auto idx : range( loops ) )
    {
        prediction = autoencoder.predict( prediction );
        {
            auto data = prediction;
            std::string file_name = std::string{"./tmp/mnist_autoencoder_"} + std::to_string( idx ) + std::string{".png"};
            imageio::imwrite( file_name, data.reshape( {28, 28} ) );
        }
    }

    return 0;
}

