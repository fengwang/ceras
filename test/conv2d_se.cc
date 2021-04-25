#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include "../include/utils/color.hpp"
#include <cmath>
#include <iostream>

void test( unsigned long output_channels,std::vector<unsigned long> const& kernel_size, std::vector<unsigned long> const& input_shape, std::string const& padding="valid", std::vector<unsigned long> const& strides={1,1} )
{
    unsigned long batch_size = 16;

    std::vector<unsigned long> input_data_shape;
    input_data_shape.push_back( batch_size );
    std::copy( input_shape.begin(), input_shape.end(), std::back_inserter( input_data_shape ) );
    ceras::tensor<float> input_data = ceras::random<float>( input_data_shape );

    auto input = ceras::variable{ input_data };
    auto output = ceras::Conv2D( output_channels, kernel_size, input_shape, padding, strides )(input);
    //ceras::session<ceras::tensor<double>> s;
    auto& s = ceras::get_default_session<ceras::tensor<double>>();
    auto result = s.run( output );
    auto output_shape = result.shape();

    std::cout << "For conv2d (batch size 16) with output_channels " << output_channels << " and kernel_size[0] " << kernel_size[0]
              << " and input_shape[0] " << input_shape[0] << " and padding " << padding << " and strides[0] " << strides[0] << std::endl;
    std::cout << "The output shape is :\n";
    std::copy( output_shape.begin(), output_shape.end(), std::ostream_iterator<unsigned long>{std::cout, " " } );
    std::cout << "\n";

}

int main()
{
    test( 1, {1, 1}, {1, 1, 1}, "valid", {1, 1} );
    test( 2, {1, 1}, {1, 1, 1}, "valid", {1, 1} );
    test( 1, {1, 1}, {1, 1, 1}, "same", {1, 1} );
    test( 2, {1, 1}, {1, 1, 1}, "same", {1, 1} );

    test( 1, {2, 2}, {8, 8, 1}, "valid", {1, 1} );
    test( 2, {2, 2}, {8, 8, 1}, "valid", {1, 1} );
    test( 1, {2, 2}, {8, 8, 1}, "same", {1, 1} );
    test( 2, {2, 2}, {8, 8, 1}, "same", {1, 1} );

    test( 1, {2, 2}, {8, 8, 1}, "valid", {2, 2} );
    test( 2, {2, 2}, {8, 8, 1}, "valid", {2, 2} );
    test( 1, {2, 2}, {8, 8, 1}, "same", {2, 2} );
    test( 2, {2, 2}, {8, 8, 1}, "same", {2, 2} );

    test( 1, {3, 3}, {16, 16, 1}, "valid", {1, 1} );
    test( 2, {3, 3}, {16, 16, 1}, "valid", {1, 1} );
    test( 1, {3, 3}, {16, 16, 1}, "same", {1, 1} );
    test( 2, {3, 3}, {16, 16, 1}, "same", {1, 1} );

    test( 1, {3, 3}, {16, 16, 1}, "valid", {2, 2} );
    test( 2, {3, 3}, {16, 16, 1}, "valid", {2, 2} );
    test( 1, {3, 3}, {16, 16, 1}, "same", {2, 2} );
    test( 2, {3, 3}, {16, 16, 1}, "same", {2, 2} );

    test( 1, {4, 4}, {16, 16, 1}, "valid", {1, 1} );
    test( 2, {4, 4}, {16, 16, 1}, "valid", {1, 1} );
    test( 1, {4, 4}, {16, 16, 1}, "same", {1, 1} );
    test( 2, {4, 4}, {16, 16, 1}, "same", {1, 1} );

    test( 1, {4, 4}, {16, 16, 1}, "valid", {2, 2} );
    test( 2, {4, 4}, {16, 16, 1}, "valid", {2, 2} );
    test( 1, {4, 4}, {16, 16, 1}, "same", {2, 2} );
    test( 2, {4, 4}, {16, 16, 1}, "same", {2, 2} );


    test( 1, {5, 5}, {16, 16, 1}, "valid", {1, 1} );
    test( 2, {5, 5}, {16, 16, 1}, "valid", {1, 1} );
    test( 1, {5, 5}, {16, 16, 1}, "same", {1, 1} );
    test( 2, {5, 5}, {16, 16, 1}, "same", {1, 1} );

    test( 1, {5, 5}, {16, 16, 1}, "valid", {2, 2} );
    test( 2, {5, 5}, {16, 16, 1}, "valid", {2, 2} );
    test( 1, {5, 5}, {16, 16, 1}, "same", {2, 2} );
    test( 2, {5, 5}, {16, 16, 1}, "same", {2, 2} );
    return 0;
}

