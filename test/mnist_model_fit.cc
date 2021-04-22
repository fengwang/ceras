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
    using namespace ceras;
    auto input = Input();
    auto l1 = relu( Dense( 256, 28*28 )( input ) );
    auto l2 = sigmoid( Dense( 128, 256 )( l1 ) );
    auto output = Dense( 10, 128 )( l2 );
    return model( input, output );
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
    float learning_rate = 0.01f;
    auto cm = m.compile( CategoricalCrossentropy(), SGD(batch_size, learning_rate) );

    unsigned long epoches = 10;
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
            std::cout << "Prediction error at " << i << ": predicted " << predicted_number << ", but the ground_truth is " << ground_truth <<  "\r" << std::flush;
        }

    }

    float const err = 1.0 * errors / 10000;
    std::cout << "\nPrediction error on the test set is " << err << std::endl;

}

int main()
{
    auto m = train_model();
    evaluate( m );
    return 0;
}

