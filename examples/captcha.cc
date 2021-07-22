#include "../include/ceras.hpp"
#include "../include/utils/3rd_party/glob.hpp"
#include "../include/utils/imageio.hpp"

#include <iostream>

// finding the index for characters in sequence '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
inline unsigned long find_index( char ch )
{
    constexpr std::array<unsigned long, 256> table{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };

    return table[static_cast<unsigned long>(ch)];
}

// loading dataset from a path, this folder is full of png files with names such as '00000000_TSCZ.png' -- '00000000' is for the index, and 'TSCZ' is for the characters inside this image
// this function will generate a dataset, giving the [images, first character, second character, third character]
// example usage: auto [imgs, l_0, l_1, l_2, l_3 ] = load_dataset( "./dataset/captcha/training/*.png" );
inline auto load_dataset( std::string const& path )
{
    using namespace ceras;

    std::vector<std::filesystem::path> image_paths = glob::glob( path );
    unsigned long const N = image_paths.size();
    std::cout << "Loading dataset... Found " << N << "images from path " << path << std::endl;

    tensor<float> images{ {N, 80, 160, 3} };
    auto label_0 = zeros<float>( {N, 36} );
    auto label_1 = zeros<float>( {N, 36} );
    auto label_2 = zeros<float>( {N, 36} );
    auto label_3 = zeros<float>( {N, 36} );

    view_2d<float> l0{ label_0.data(), N, 36 };
    view_2d<float> l1{ label_1.data(), N, 36 };
    view_2d<float> l2{ label_2.data(), N, 36 };
    view_2d<float> l3{ label_3.data(), N, 36 };

    for ( auto idx : range( N ) )
    {
        //
        std::filesystem::path const& p = image_paths[idx];
        tensor<unsigned char> const& img = imageio::imread( p.c_str() );
        std::transform( img.begin(), img.end(), images.begin()+idx*80*160*3, []( unsigned char ch ){ return static_cast<float>(static_cast<int>(ch)) / 255.0f; } );
        // 00000000_TSCZ.png
        std::string const file_name = p.filename().c_str();
        std::string label_4 = file_name.substr( 9, 4 );
        l0[idx][find_index(label_4[0])] = 1.0f;
        l1[idx][find_index(label_4[1])] = 1.0f;
        l2[idx][find_index(label_4[2])] = 1.0f;
        l3[idx][find_index(label_4[3])] = 1.0f;
    }

    return std::make_tuple( images, label_0, label_1, label_2, label_3 );
}


int main()
{
    using namespace ceras;
    random_generator.seed( 42 );

    // model
    auto input = Input(); // shape( 80, 160, 3 )
    auto l1 = ReLU(BatchNormalization({80, 160, 8})(Conv2D( 8, {3, 3}, {80, 160, 3}, "same" )( input )));
    auto l2 = MaxPooling2D( 2 )( l1 );
    auto l2 = ReLU(BatchNormalization({40, 80, 16})(Conv2D( 16, {3, 3}, {40, 80, 8}, "same" )( l2 )));
    auto l3 = MaxPooling2D( 2 )( l2 );
    auto l4 = ReLU(BatchNormalization({20, 40, 32})(Conv2D( 32, {3, 3}, {20, 40, 16}, "same" )( l3 )));
    auto l5 = MaxPooling2D( 2 )( l4 );
    auto l6 = ReLU(BatchNormalization({10, 20, 64})(Conv2D( 256, {3, 3}, {10, 20, 32}, "same" )( l5 )));
    auto l7 = MaxPooling2D( 2 )( l6 );
    auto l8 = Flatten()( l7 ); // 5x10x64
    auto l9 = ReLU( Dense( 1024, 3200 )( l8 ) );

    auto out_0 = Dense( 36, 1024 )( l9 );
    auto out_1 = Dense( 36, 1024 )( l9 );
    auto out_2 = Dense( 36, 1024 )( l9 );
    auto out_3 = Dense( 36, 1024 )( l9 );


    // losses
    auto gt_out_0 = place_holder<tensor<float>>{};
    auto gt_out_1 = place_holder<tensor<float>>{};
    auto gt_out_2 = place_holder<tensor<float>>{};
    auto gt_out_3 = place_holder<tensor<float>>{};
    auto loss_0 = cross_entropy_loss( gt_out_0, out_0 );
    auto loss_1 = cross_entropy_loss( gt_out_1, out_1 );
    auto loss_2 = cross_entropy_loss( gt_out_2, out_2 );
    auto loss_3 = cross_entropy_loss( gt_out_3, out_3 );

    auto combined_loss = loss_0 + loss_1 + loss_2 + loss_3; // only when all losses of same shape

    // dataset
    unsigned long const batch_size = 256;
    tensor<float> images{ {batch_size, 80, 160, 3} };
    tensor<float> labels_0{ {batch_size, 36} };
    tensor<float> labels_1{ {batch_size, 36} };
    tensor<float> labels_2{ {batch_size, 36} };
    tensor<float> labels_3{ {batch_size, 36} };

    // session
    auto& s = get_default_session<tensor<float>>();
    s.bind( input, images );
    s.bind( gt_out_0, labels_0 );
    s.bind( gt_out_1, labels_1 );
    s.bind( gt_out_2, labels_2 );
    s.bind( gt_out_3, labels_3 );


    // optimizer
    unsigned long const epochs = 32;
    float learning_rate = 0.05f;
    auto opt_0 = gradient_descent{ loss_0, batch_size, learning_rate };
    auto opt_1 = gradient_descent{ loss_1, batch_size, learning_rate };
    auto opt_2 = gradient_descent{ loss_2, batch_size, learning_rate };
    auto opt_3 = gradient_descent{ loss_3, batch_size, learning_rate };


    // load training dataset
    auto& [imgs, l_0, l_1, l_2, l_3 ] = load_dataset( "./dataset/captcha/training/*.png" );
    unsigned long const N = *(imgs.shape().begin());
    unsigned long iterations = N / batch_size;

    for ( [[maybe_unused]] auto e : range(epoches) )
    {
        for ( auto i : range( iterations ) )
        {
            // prepare data
            auto loss = s.run( combined_loss );

            //
            s.run( opt_0 );
            s.run( opt_1 );
            s.run( opt_2 );
            s.run( opt_3 );
        }
    }




    std::size_t const batch_size = 10;
    float learning_rate = 0.005f;
    auto cm = m.compile( CategoricalCrossentropy(), SGD(batch_size, learning_rate) );

    unsigned long epoches = 10;
    int verbose = 1;
    double validation_split = 0.1;
    auto const& [x_training, y_training, x_test, y_test] = dataset::mnist::load_data();

    auto history = cm.fit( x_training.as_type<float>()/255.0f, y_training.as_type<float>(), batch_size, epoches, verbose, validation_split );

    auto error = cm.evaluate( x_test.as_type<float>()/255.0, y_test.as_type<float>(), batch_size );

    std::cout << "\nPrediction error on the test set is " << error << std::endl;

    return 0;
}

