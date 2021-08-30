#include "../../include/ceras.hpp"
#include "../../include/utils/range.hpp"
#include "../../include/utils/better_assert.hpp"

#include <fstream>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <iterator>


using namespace ceras;
typedef tensor<float> tensor_type;

int main()
{
    auto input = Input(); //  3D tensor input, (batch_size, 224, 224, 3)
    auto l0 = relu( Conv2D( 64, {3, 3}, {224, 224, 3}, "same" )(input) ); // 224, 224, 64
    auto l1 = max_pooling_2d( 2 ) ( l0 ); // 112, 112, 64
    auto l2 = relu( Conv2D( 128, {3, 3}, {112, 112, 64}, "same" )( l1 ) ); // 112, 112, 128
    auto l3 = relu( Conv2D( 128, {3, 3}, {112, 112, 128}, "same" )( l2 ) ); // 112, 112, 128
    auto l4 = max_pooling_2d( 2 ) ( l3 ); // 56, 56, 128
    auto l5 = relu( Conv2D( 256, {3, 3}, {56, 56, 128}, "same" )( l4 ) ); // 56, 56, 256
    auto l6 = relu( Conv2D( 256, {3, 3}, {56, 56, 256}, "same" )( l5 ) ); // 56, 56, 256
    auto l7 = relu( Conv2D( 256, {3, 3}, {56, 56, 256}, "same" )( l6 ) ); // 56, 56, 256
    auto l8 = max_pooling_2d( 2 ) ( l7 ); // 28, 28, 256
    auto l9 = relu( Conv2D( 512, {3, 3}, {28, 28, 256}, "same" )( l8 ) ); // 28, 28, 512
    auto l10 = relu( Conv2D( 512, {3, 3}, {28, 28, 512}, "same" )( l9 ) ); // 28, 28, 512
    auto l11 = relu( Conv2D( 512, {3, 3}, {28, 28, 512}, "same" )( l10 ) ); // 28, 28, 512
    auto l12 = max_pooling_2d( 2 ) ( l11 ); // 14, 14, 512
    auto l13 = relu( Conv2D( 512, {3, 3}, {14, 14, 512}, "same" )( l12 ) ); // 14, 14, 512
    auto l14 = relu( Conv2D( 512, {3, 3}, {14, 14, 512}, "same" )( l13 ) ); // 14, 14, 512
    auto l15 = relu( Conv2D( 512, {3, 3}, {14, 14, 512}, "same" )( l14 ) ); // 14, 14, 512
    auto l16 = max_pooling_2d( 2 ) ( l15 ); // 7, 7, 512
    auto l17 = flatten( l16 ); // 7x7x512
    auto l18 = relu( Dense( 4096, 7*7*512 )( l17 ) ); // 4096
    auto l19 = relu( Dense( 4096, 4096 )( l18 ) ); // 4096
    auto l20 = identity( Dense( 1000, 4096 )( l19 ) ); // 1000
    auto output = l20;

    auto m = model{ input, output }; // define a model
    m.summary( "./examples/vgg16/vgg16.dot" );
    //m.save_weights( "./examples/vgg16/vgg16.weights" ); // <- slow lzw compression, need optimizing


    auto ground_truth = place_holder<tensor_type>{}; // 1-D, 1000
    auto loss = cross_entropy_loss( ground_truth, output );

    #if 0
    training code ommited.
    #endif

    return 0;
}

