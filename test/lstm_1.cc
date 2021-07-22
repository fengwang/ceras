#include "../include/ceras.hpp"
#include "../include/recurrent_operation.hpp"
#include "../include/utils/range.hpp"
#include "../include/utils/better_assert.hpp"
#include "../include/utils/color.hpp"

#include <fstream>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <cmath>


// generating dataset with noise enabled
ceras::tensor<float> generate_dataset( unsigned long N, float B = 2.7f, float w = 0.025f, float fs = 2.0f, float theta = 2.3f, float phi = 1.5f, float noise_amplitude=0.1f )
{
    ceras::tensor<float> ans{ {N,} };
    for ( auto idx : ceras::range(N) )
        *(ans.data()+idx) = std::cos(B*std::cos(w*idx/fs+theta)+phi);

    ceras::tensor<float> noise = ceras::random( {N,}, -noise_amplitude, noise_amplitude );
    return ans + noise;
}

int main()
{
    ceras::random_generator.seed( 42 );

    using namespace ceras;
    typedef tensor<float> tensor_type;

    constexpr std::size_t const batch_size = 1024;
    constexpr std::size_t const iteration_per_epoch = 1024;
    constexpr unsigned long N = (batch_size+1)*(iteration_per_epoch+1);
    float learning_rate = 1.0e-1f;

    auto const& dataset_ = generate_dataset( N );
    auto&& dataset = view_2d<float>{ dataset_.data(), batch_size+1, iteration_per_epoch+1 };

    auto input = place_holder<tensor_type>{}; // 1-D, 1 pixels

    auto ls_1 = lstm( 1, 32 )( input ); // lstm layer 1

    auto w = variable{ glorot_uniform<float>( {32, 1} ) };
    auto b = variable{ zeros<float>( {1,} ) };
    auto output = tanh( ls_1 * w + b );

    auto ground_truth = place_holder<tensor_type>{}; // 1-D, 1 pixels
    auto loss = mse( ground_truth, output );

    tensor_type input_d{ {batch_size, 1} };
    tensor_type output_d{ {batch_size, 1} };

    std::size_t const epoch = 2;

    // creating session
    auto& s = get_default_session<tensor_type>();
    s.bind( input, input_d );
    s.bind( ground_truth, output_d );

    // proceed training
    auto optimizer = gradient_descent{ loss, batch_size, learning_rate };

    for ( auto e : range( epoch ) )
    {

        for ( auto i : range( iteration_per_epoch ) )
        {
            for ( auto j : range( batch_size ) )
            {
                input_d[j] = dataset[j][i];
                output_d[j] = dataset[j][i+1];
            }

            auto current_error = s.run( loss );
            std::cout << "Loss at epoch " << e << " index: " << (i+1)*batch_size << ":\t" << current_error[0] << "\r" << std::flush;
            better_assert( !has_nan(current_error), "Error in current loss." );
            s.run( optimizer );
        }
        std::cout << std::endl;

        output.reset_states();
    }

    std::cout << std::endl;


    /*
    unsigned long const new_batch_size = 1;

    std::vector<std::uint8_t> testing_images = load_binary( testing_image_path );
    std::vector<std::uint8_t> testing_labels = load_binary( testing_label_path );
    std::size_t const testing_iterations = 10000 / new_batch_size;

    tensor<float> new_input{ {new_batch_size, 28 * 28} };
    s.bind( input, new_input );

    unsigned long errors = 0;

    for ( auto i = 0UL; i != testing_iterations; ++i )
    {
        std::size_t const image_offset = 16 + i * new_batch_size * 28 * 28;

        for ( auto j = 0UL; j != new_batch_size*28*28; ++j )
            new_input[j] = static_cast<float>( testing_images[j + image_offset] ) / 127.5f - 1.0f;

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
    */

    return 0;
}

