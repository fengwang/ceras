#include "../../include/ceras.hpp"
#include "../../include/utils/range.hpp"
#include "../../include/utils/better_assert.hpp"


int main()
{
    using namespace ceras;
    auto input = Input(); //  3D tensor input, (batch_size, 256, 256, 3)

    auto l0 = relu( Conv2D( 64, {3, 3}, {256, 256, 3}, "same" )(input) ); // 256, 256, 64
    auto l1 = max_pooling_2d( 2 ) ( l0 ); // 128, 128, 64

    auto l2 = relu( Conv2D( 128, {3, 3}, {128, 128, 64}, "same" )( l1 ) ); // 128, 128, 128
    auto l3 = relu( Conv2D( 128, {3, 3}, {128, 128, 128}, "same" )( l2 ) ); // 128, 128, 128
    auto l4 = max_pooling_2d( 2 ) ( l3 ); // 64, 64, 128

    auto l5 = relu( Conv2D( 256, {3, 3}, {64, 64, 128}, "same" )( l4 ) ); // 64, 64, 256
    auto l6 = relu( Conv2D( 256, {3, 3}, {64, 64, 256}, "same" )( l5 ) ); // 64, 64, 256
    auto l7 = relu( Conv2D( 256, {3, 3}, {64, 64, 256}, "same" )( l6 ) ); // 64, 64, 256
    auto l8 = max_pooling_2d( 2 ) ( l7 ); // 32, 32, 256

    auto l9 = relu( Conv2D( 512, {3, 3}, {32, 32, 256}, "same" )( l8 ) ); // 32, 32, 512
    auto l10 = relu( Conv2D( 512, {3, 3}, {32, 32, 512}, "same" )( l9 ) ); // 32, 32, 512
    auto l11 = relu( Conv2D( 512, {3, 3}, {32, 32, 512}, "same" )( l10 ) ); // 32, 32, 512
    auto l12 = max_pooling_2d( 2 ) ( l11 ); // 16, 16, 512

    auto l13 = relu( Conv2D( 512, {3, 3}, {16, 16, 512}, "same" )( l12 ) ); // 16, 16, 512
    auto l14 = relu( Conv2D( 512, {3, 3}, {16, 16, 512}, "same" )( l13 ) ); // 16, 16, 512
    auto l15 = relu( Conv2D( 512, {3, 3}, {16, 16, 512}, "same" )( l14 ) ); // 16, 16, 512
    auto l16 = max_pooling_2d( 2 ) ( l15 ); // 8, 8, 512

    auto l17 = relu( Conv2D( 512, {3, 3}, {8, 8, 512}, "same" )( l16 ) ); // 8, 8, 512
    auto l18 = relu( Conv2D( 512, {3, 3}, {8, 8, 512}, "same" )( l17 ) ); // 8, 8, 512

    auto l19 = up_sampling_2d( 2 )( l18 ); // 16, 16, 512
    auto l20 = l15 + relu( Conv2D( 512, {3, 3}, {16, 16, 512}, "same" )( l19 ) ); // or concatenate instead of '+'

    auto l21 = relu( Conv2D( 512, {3, 3}, {16, 16, 512}, "same" )( l20 ) ); // 16, 16, 512
    auto l22 = relu( Conv2D( 512, {3, 3}, {16, 16, 512}, "same" )( l21 ) ); // 16, 16, 512

    auto l23 = up_sampling_2d(2)( l22 ); // 32, 32, 512
    auto l24 = l11 + relu( Conv2D( 512, {3, 3}, {32, 32, 512}, "same" )( l23 ) ); // 32, 32, 512

    auto l25 = relu( Conv2D( 512, {3, 3}, {32, 32, 512}, "same" )( l24 ) );
    auto l26 = relu( Conv2D( 512, {3, 3}, {32, 32, 512}, "same" )( l25 ) );

    auto l27 = up_sampling_2d(2)( l26 ); // 64, 64, 512
    auto l28 = l7 + relu( Conv2D( 256, {3, 3}, {64, 64, 512}, "same" )( l27 ) ); //64, 64, 256

    auto l29 = relu( Conv2D( 256, {3, 3}, {64, 64, 256}, "same" )( l28 ) ); // 64, 64, 256
    auto l30 = relu( Conv2D( 256, {3, 3}, {64, 64, 256}, "same" )( l29 ) ); // 64, 64, 256

    auto l31 = up_sampling_2d(2)( l30 ); // 128, 128, 256
    auto l32 = l3 + relu( Conv2D( 128, {3, 3}, {128, 128, 256}, "same" )( l31 ) );

    auto l33 = relu( Conv2D( 128, {3, 3}, {128, 128, 256}, "same" )( l32 ) );
    auto l34 = relu( Conv2D( 128, {3, 3}, {128, 128, 128}, "same" )( l33 ) ); // 128, 128, 128

    auto l35 = up_sampling_2d(2)( l34 ); // 256, 256, 128
    auto l36 = relu( Conv2D( 64, {3, 3}, {256, 256, 128}, "same" )( l35 ) );
    auto l37 = sigmoid( Conv2D( 3, {3, 3}, {256, 256, 128}, "same" )( l36 ) );

    auto output = l37;

    auto m = model{ input, output }; // define a model
    m.summary( "./examples/unet/unet.dot" );

    //training code ommited.

    return 0;
}

