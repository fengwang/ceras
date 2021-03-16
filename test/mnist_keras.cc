#include "../include/ceras.hpp"
#include "../include/utils/range.hpp"
#include "../include/utils/better_assert.hpp"
#include "../include/utils/color.hpp"
#include "../include/keras/layer.hpp"
#include "../include/keras/model.hpp"


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

/*
std::size_t const image_offset = 16 + i * batch_size * 28 * 28;
for ( auto j : range( batch_size * 28 * 28 ) )
    input_images[j] = static_cast<float>(training_images[j+image_offset]) / 127.5f - 1.0f;
*/
ceras::tensor<float> process_images( std::vector<std::uint8_t> const& images ) // [uint8, ..., uint8] --> [ dims, 28*28 ]
{
    unsigned long const offset = 16;
    unsigned long const size = images.size()-offset;
    unsigned long const samples = size / (28*28);
    auto zeros = ceras::zeros<float>( {samples, 28*28} );
    ceras::for_each( zeros.begin(), zeros.end(), images.begin()+offset, []( float& x, std::uint8_t v ) { x = static_cast<float>(v) / 127.5f - 1.0f; } );
    return zeros;
}
ceras::tensor<float> process_labels( std::vector<std::uint8_t> const& labels ) // [uint8, ..., uint8] --> [ dims, 10 ]
{
    unsigned long const offset = 8;
    unsigned long const size = labels.size()-offset;
    unsigned long const samples = size;
    auto zeros = ceras::zeros<float>( {samples, 10} );
    auto _zeros = ceras::view_2d{zeros, samples, 10};
    for ( auto idx : ceras::range( samples ) )
    {
        unsigned long const label = static_cast<unsigned long>(labels[idx+offset]);
        _zeros[idx][label] = 1.0f;
    }
    return zeros;
}

/*
    using namespace Keras;
    auto input = Input( {28*28,} );
    auto layer_1 = Dense<512, activation<"relu">, use_bias<false>>{}( input );
    auto layer_2 = Dense<128, activation<"leaky_relu">>{}( layer_1 );
    auto layer_3 = Dense<32, activation<"relu">>{}( layer_2 );
    auto layer_4 = Dense<10>{}( layer_3 );

    auto model = Model{ input, layer_4 };
    //auto compiled_model = model.compile<loss<"crossentropy">, optimizer<"sgd", 32, "0.08">>();
    auto compiled_model = model.compile<optimizer<"sgd", 32, "0.08">, loss<"crossentropy">>();

    auto fake_inputs = ceras::random<float>( {32, 28*28} );
    auto fake_outputs = ceras::ones<float>( {32, 10} );

    auto error = compiled_model.train_on_batch( fake_inputs, fake_outputs );

    std::cout << "Got error :\n"  << error << "\n";

*/
int main()
{
    ceras::random_generator.seed( 42 );


    // prepare dataset
    std::vector<std::uint8_t> training_images = load_binary( training_image_path ); // [u32, u32, u32, u32, uint8, uint8, ... ]
    auto training_input = process_images( training_images );
    std::vector<std::uint8_t> training_labels = load_binary( training_label_path ); // [u32, u32, uint8, uint8, ... ]
    auto training_output = process_labels( training_labels );

    std::vector<std::uint8_t> testing_images = load_binary( testing_image_path );
    auto testing_input = process_images( training_images );
    std::vector<std::uint8_t> testing_labels = load_binary( testing_label_path );
    auto testing_output = process_labels( training_labels );


    // define model
    using namespace Keras;
    auto input = Input( {28*28} );
    auto l_1 = Dense<256, activation<"relu">, use_bias<false>>{}( input );
    auto l_2 = Dense<128, activation<"relu">>{}( l_1 );
    auto l_3 = Dense<10, activation<"relu">>{}( l_2 );
    auto model = Model{ input, l_3 };

    // training
    //auto compiled_model = model.compile<optimizer<"sgd", 10, "0.1">, loss<"crossentropy">>();
    auto compiled_model = model.compile<optimizer<"sgd", 1000, "0.1">, loss<"crossentropy">>();
    unsigned long const batch_size = 1000;
    unsigned long const epochs = 1;
    auto terrors = compiled_model.fit( training_input, training_output, batch_size, epochs);
    std::cout << "\nTraining errors:\n";
    for ( auto error : terrors ) std::cout << error << " ";
    std::cout << "\n";

    // prediction
    auto prediction = compiled_model.predict( testing_input );
    unsigned long const test_cases = *((prediction.shape()).begin());

    unsigned long errors = 0;
    for ( auto idx : ceras::range( test_cases ) )
    {
        unsigned long predicted_number = std::max_element( prediction.begin()+10*idx, prediction.begin()+10*idx+10 ) - prediction.begin() - 10*idx;
        unsigned long ground_truth_number = std::max_element( testing_output.begin()+10*idx, testing_output.begin()+10*idx+10 ) - testing_output.begin() - 10*idx;
        errors = (predicted_number == ground_truth_number) ? errors : errors + 1;
    }

    float const err = 1.0 * errors / test_cases;
    std::cout << "Prediction error on the testing set is " << err << std::endl;

    return 0;
}

