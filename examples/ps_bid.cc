#include "../include/ceras.hpp"
#include "../include/utils/range.hpp"
#include "../include/utils/better_assert.hpp"
#include "../include/utils/color.hpp"
#include "../include/utils/debug.hpp"

#include <fstream>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <iterator>

namespace
{
    template< typename T, typename ... Args >
    constexpr std::vector<typename std::remove_cvref_t<T>> _make_vector( std::vector<typename std::remove_cvref_t<T>>& ans, T const& val, Args const& ... args )
    {
        ans.push_back( val );
        if constexpr ( sizeof...(args) > 0 )
            return _make_vector( ans, args... );
        else
            return ans;
    }
}

template< typename T, typename ... Args >
constexpr std::vector<typename std::remove_cvref_t<T>> make_vector( T const& value, Args const& ... args )
{
    std::vector<typename std::remove_cvref_t<T>> ans;
    ans.reserve( sizeof...(args) + 1 );
    return _make_vector( ans, value, args... );
}

template< typename T >
constexpr std::vector<T> make_vector( std::initializer_list<T> lst )
{
    std::vector<T> ans;
    ans.resize( lst.size() );
    std::copy( std::begin(lst), std::end(lst), std::begin(ans) );
    return ans;
}

int main()
{
    ceras::random_generator.seed( 42 );
    //load training set
    // define computation graph, a 3-layered dense net with topology 784x256x128x10
    using namespace ceras;
    typedef tensor<float> tensor_type;

    unsigned long const R = 8;
    unsigned long const C = 288;

    auto qd_init = tensor_type{ make_vector(R, C) };
    {
        auto mat = view_2d{qd_init, R, C};
        // TODO: initialize here
    }
    auto Qd = variable{ qd_init };

    auto x_init = tensor_type{ make_vector(R, C) };
    {
        auto mat = view_2d{x_init, R, C};
        // TODO: initialize here
    }
    auto X = variable{ x_init };

    auto qc_init = tensor_type{ make_vector(C) };
    {
        auto mat = view_2d{qc_init, R, C};
        // TODO: initialize here
    }
    auto Qc = variable{ qc_init };

    auto m_init = tensor_type{ make_vector(C) };
    {
        auto mat = view_2d{m_init, R, C};
        // TODO: initialize here
    }
    auto M = constant{ m_init };

    auto b_init = tensor_type{ make_vector(R) };
    {
        auto mat = view_2d{b_init, R, C};
        // TODO: initialize here
    }
    auto B = variable{ b_init };

    // reward
    auto reward = sum_reduce( elementwise_product( sum_reduce(0)(elementwise_product(Qd, X))-Qc, M ) );

    float const D = 50.0f;
    float const C = 50.0f;


    // constraint

    // 1
    sum_reduce( 0 )( Qd ); // TODO: in range of [0, D/12]
    // 2
    Qc; // TODO: in range of [0, C/12]
    // 3
    auto b_coef = tensor<float>{ make_vector(R-1, R) };
    {
        auto mat = view_2d{b_coef, R-1, R};
        for ( auto r : range(R-1) )
        {
            mat[r][r] = 1.0f;
            mat[r][r+1] = -1.0f;
        }
    }
    auto B_coef = constant{ b_coef };
    sum_reduce( B_coef * reshape( make_vector(R, 1), false )( B ) );





    auto input = place_holder<tensor_type>{}; // 1-D, 28x28 pixels

    // 1st layer
    auto w1 = variable{ randn<float>( {28*28, 256}, 0.0, 10.0/(28.0*16.0) ) };
    //auto b1 = variable{ zeros<float>( { 1, 256 } ) };
    auto b1 = variable{ zeros<float>( { 256, } ) };
    auto l1 = relu( input * w1 + b1 );

    // 2nd layer
    auto w2 = variable{ randn<float>( {256, 128}, 0.0, 3.14/(16.0*11.2 )) };
    //auto b2 = variable{ zeros<float>( { 1, 128 } ) };
    auto b2 = variable{ zeros<float>( { 128, } ) };
    //auto l2 = relu( l1 * w2 + b2 );
    auto l2 = sigmoid( l1 * w2 + b2 );

    // 3rd layer
    auto w3 = variable{ randn<float>( {128, 10}, 0.0, 1.0/35.8 ) };
    //auto b3 = variable{ zeros<float>( { 1, 10 } ) };
    auto b3 = variable{ zeros<float>( { 10, } ) };
    auto output = l2 * w3 + b3;

    auto ground_truth = place_holder<tensor_type>{}; // 1-D, 10
    auto loss = cross_entropy_loss( ground_truth, output );

    // preparing training
    std::size_t const batch_size = 10;
    //std::size_t const batch_size = 10;
    //std::size_t const batch_size = 20000;
    tensor_type input_images{ {batch_size, 28*28} };
    tensor_type output_labels{ {batch_size, 10} };

    //std::size_t const epoch = 100;
    std::size_t const epoch = 2;
    std::size_t const iteration_per_epoch = 60000/batch_size;

    // creating session
    auto& s = get_default_session<tensor_type>();
    //auto& s = get_default_session<tensor_type>().get();
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

            //better_assert( false, "stop for debug" );
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    //s.save( "./mnist.session" );

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
            ceras::debug_error( "Prediction error at ", i, ": predicted ", predicted_number, ", but the ground_truth is ", ground_truth, "\n" );
            //std::cout << "Prediction error at " << i << ": predicted " << predicted_number << ", but the ground_truth is " << ground_truth << std::endl;
        }

    }

    float const err = 1.0 * errors / 10000;
    std::cout << "Prediction error on the testing set is " << err << std::endl;

    return 0;
}

