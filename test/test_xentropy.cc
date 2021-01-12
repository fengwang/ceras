#include "../include/utils/onehot_precision.hpp"
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

int main()
{
    constexpr unsigned long dim = 10;
    constexpr unsigned long cat = 10;

    using namespace ceras;
    auto input = place_holder<tensor<double>>{};

    // 1st layer
    auto w1 = variable{ ones<double>( {dim*dim, 16} ) };
    auto b1 = variable{ zeros<double>( { 1, 16 } ) };
    auto l1 = relu( input * w1 + b1 );

    // 3rd layer
    auto w3 = variable{ ones<double>( {16, cat} ) };
    auto b3 = variable{ zeros<double>( { 1, cat } ) };
    auto output = softmax(l1 * w3 + b3);

    auto ground_truth = place_holder<tensor<double>>{}; // 1-D, 10
    auto loss = cross_entropy( ground_truth, output );
    //auto loss = cross_entropy_error( ground_truth, output );

    // preparing training
    std::size_t const batch_size = 10;
    tensor<double> input_images = zeros<double>( {batch_size, dim*dim} );
    for ( auto idx : range( batch_size ) )
    {
        unsigned long N = dim;
        unsigned long offset = idx * dim * dim + idx * dim;
        std::fill_n( input_images.begin()+offset, N, 1.0 );
    }

    tensor<double> output_labels = zeros<double>( {batch_size, cat} );
    for ( auto idx : range( batch_size ) )
        output_labels[idx*10+idx] = 1.0;

    std::size_t const epoch = 1;
    std::size_t const iteration_per_epoch = 1;

    // creating session
    session<tensor<double>> s;
    s.bind( input, input_images );
    s.bind( ground_truth, output_labels );

    // proceed training
    double learning_rate = 0.1;
    auto optimizer = gradient_descent{ loss, learning_rate };

    for ( auto e : range( epoch ) )
    {

        for ( auto i : range( iteration_per_epoch ) )
        {
            auto current_error = s.run( loss );
            std::cout << "Loss at epoch " << e << " index: " << (i+1)*batch_size << ":\t" << current_error[0] << "\n";
            better_assert( !has_nan(current_error), "Error in current loss." );
            s.run( optimizer );
        }
    }

    auto const& prediction = s.run( output );
    double precision = onehot_precision( output_labels, prediction );
    std::cout << "precision:" << precision << std::endl;


    return 0;
}

