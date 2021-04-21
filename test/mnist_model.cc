#include "../include/ceras.hpp"
#include "../include/layer.hpp"
#include "../include/utils/range.hpp"
#include "../include/utils/better_assert.hpp"
#include "../include/utils/color.hpp"
#include "../include/utils/tqdm.hpp"
#include "../include/model.hpp"

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

auto train_model()
{
    ceras::random_generator.seed( 42 );
    //load training set
    std::vector<std::uint8_t> training_images = load_binary( training_image_path ); // [u32, u32, u32, u32, uint8, uint8, ... ]
    std::vector<std::uint8_t> training_labels = load_binary( training_label_path ); // [u32, u32, uint8, uint8, ... ]


    // define computation graph, a 3-layer dense net with topology 784x256x128x10
    using namespace ceras;
    typedef tensor<float> tensor_type;
    auto input = place_holder<tensor_type>{}; // 1-D, 28x28 pixels

    auto l1 = relu( Dense( 256, 28*28 )( input ) );

    auto l2 = sigmoid( Dense( 128, 256 )( l1 ) );

    auto output = Dense( 10, 128 )( l2 );

    auto ground_truth = place_holder<tensor_type>{}; // 1-D, 10
    auto loss = cross_entropy_loss( ground_truth, output );

    // preparing training
    std::size_t const batch_size = 10;
    tensor_type input_images{ {batch_size, 28*28} };
    tensor_type output_labels{ {batch_size, 10} };

    std::size_t const epoch = 10;
    std::size_t const iteration_per_epoch = 60000/batch_size;

    // creating session
    session<tensor_type> s;
    s.bind( input, input_images );
    s.bind( ground_truth, output_labels );

    // proceed training
    float learning_rate = 1.0e-1f;
    auto optimizer = gradient_descent{ loss, batch_size, learning_rate };

    for ( [[maybe_unused]] auto e : range( epoch ) )
    {
        for ( [[maybe_unused]] auto i : tq::trange( iteration_per_epoch ) )
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
            //std::cout << "Loss at epoch " << e << " index: " << (i+1)*batch_size << ":\t" << current_error[0] << "\r" << std::flush;
            better_assert( !has_nan(current_error), "Error in current loss." );
            s.run( optimizer );
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    return model{ input, output };
}

template< typename Model >
void evaluate( Model m )
{
    using  namespace ceras;
    unsigned long const new_batch_size = 1;
    std::vector<std::uint8_t> testing_images = load_binary( testing_image_path );
    std::vector<std::uint8_t> testing_labels = load_binary( testing_label_path );
    std::size_t const testing_iterations = 10000 / new_batch_size;

    tensor<float> new_input_images{ {new_batch_size, 28 * 28} };

    //session<tensor<float> > s;
    //s.bind( input, new_input_images );

    unsigned long errors = 0;

    for ( auto i = 0UL; i != testing_iterations; ++i )
    {
        std::size_t const image_offset = 16 + i * new_batch_size * 28 * 28;

        for ( auto j = 0UL; j != new_batch_size*28*28; ++j )
            new_input_images[j] = static_cast<float>( testing_images[j + image_offset] ) / 127.5f - 1.0f;

        //auto prediction = s.run( output );
        auto prediction = m.predict( new_input_images );
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

}

int main()
{
    auto m = train_model();
    evaluate( m );
    return 0;
}

