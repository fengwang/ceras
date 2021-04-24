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

auto build_model()
{
#if 1
    using namespace ceras;
    auto input = Input();
    auto output = Dense( 10, 28*28 )( input );
    return model( input, output );
#else
    using namespace ceras;
    auto input = Input();
    auto l1 = relu( Dense( 256, 28*28 )( input ) );
    auto l2 = sigmoid( Dense( 128, 256 )( l1 ) );
    auto output = Dense( 10, 128 )( l2 );
    return model( input, output );
#endif
}

auto train_model()
{
    using namespace ceras;
    typedef tensor<float> tensor_type;
    random_generator.seed( 42 );

    //load training set
    std::vector<std::uint8_t> training_images = load_binary( training_image_path ); // [u32, u32, u32, u32, uint8, uint8, ... ]
    std::vector<std::uint8_t> training_labels = load_binary( training_label_path ); // [u32, u32, uint8, uint8, ... ]

    // preparing training data
    unsigned long samples = 60000;
    tensor_type input_data{ {samples, 28*28} };
    for_each( training_images.begin()+16, training_images.end(), input_data.begin(), []( std::uint8_t x, float& v){ v = 1.0f * x / 127.5f - 1.0f; } );

    tensor_type output_data{ {samples, 10} } ;
    std::fill( output_data.begin(), output_data.end(), 0.0f );
    for ( auto idx : range( samples ) )
    {
        std::size_t const label = training_labels[8+idx];
        output_data[idx*10+label] = 1.0f;
    }

    auto m = build_model();
    std::size_t const batch_size = 10;
    //float learning_rate = 0.01f;
    float learning_rate = 0.005f;
    auto cm = m.compile( CategoricalCrossentropy(), SGD(batch_size, learning_rate) );

    unsigned long epoches = 150;
    int verbose = 1;
    double validation_split = 0.1;
    auto history = cm.fit( input_data, output_data, batch_size, epoches, verbose, validation_split );

    auto const& [training_loss, validation_loss] = history;
    for ( auto idx : range( epoches ) )
    {
        std::cout << training_loss[idx] << " -- " << validation_loss[idx] << std::endl;
    }

    return cm;
}

template< typename Model >
void evaluate( Model m )
{
    using namespace ceras;
    typedef tensor<float> tensor_type;
    std::vector<std::uint8_t> testing_images = load_binary( testing_image_path );
    std::vector<std::uint8_t> testing_labels = load_binary( testing_label_path );

    unsigned long const samples = 10000;
    tensor_type input_data{ {samples, 28*28} };
    for_each( testing_images.begin()+16, testing_images.end(), input_data.begin(), []( std::uint8_t x, float& v){ v = 1.0f * x / 127.5f - 1.0f; } );

    tensor_type output_data{ {samples, 10} } ;
    std::fill( output_data.begin(), output_data.end(), 0.0f );
    for ( auto idx : range( samples ) )
    {
        std::size_t const label = testing_labels[8+idx];
        output_data[idx*10+label] = 1.0f;
    }

    std::size_t const batch_size = 10;
    auto error = m.evaluate( input_data, output_data, batch_size );

    std::cout << "\nPrediction error on the test set is " << error << std::endl;

}

int main()
{
    auto m = train_model();
    evaluate( m );
    return 0;
}

