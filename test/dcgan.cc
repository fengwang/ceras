#include "../include/ceras.hpp"
#include "../include/utils/imageio.hpp"
#include <iostream>

//using namespace ceras;
auto build_generator( unsigned long const latent_dim )
{
    auto input = ceras::Input(); // (latent_dim, )
    auto l1 = ceras::ReLU( ceras::Dense( 128*7*7, latent_dim )( input ) );
    auto l2 = ceras::Reshape( {7, 7, 128} )( l1 );
    auto l3 = ceras::UpSampling2D( 2 )( l2 ); // (14, 14, 128)
    auto l4 = ceras::Conv2D( 128, {3, 3}, {14, 14, 128}, "same" )( l3 );
    auto l5 = ceras::ReLU( ceras::BatchNormalization( 0.8f, {14, 14, 128} )( l4 ) );
    auto l6 = ceras::UpSampling2D( 2 )( l5 ); // (28, 28, 128)
    auto l7 = ceras::Conv2D( 64, {3, 3}, {28, 28, 128}, "same" )( l6 );
    auto l8 = ceras::ReLU( ceras::BatchNormalization( 0.8f, {28, 28, 64} )( l7 ) );
    auto output = ceras::tanh( ceras::Conv2D( 1, {3,3}, {28, 28, 64}, "same" )( l8 ) );
    return ceras::model{ input, output };
}

auto build_discriminator()
{
    auto input = ceras::Input(); // (28, 28, 1)
    auto l1 = ceras::MaxPooling2D(2)( ceras::Dropout(0.25f)( ceras::LeakyReLU(0.2f)( ceras::Conv2D( 32, {3, 3}, {28, 28, 1}, "same" )( input ) ) ) ); // (14, 14, 32)
    auto l2 = ceras::MaxPooling2D(2)( ceras::Dropout(0.25f)( ceras::LeakyReLU(0.2f)( ceras::BatchNormalization(0.8f, {14, 14, 64})( ceras::LeakyReLU(0.2f)( ceras::Conv2D( 64, {3, 3}, {14, 14, 32}, "same" )( l1 ) ) ) ) ) ); //(7, 7, 64)
    auto l3 = ceras::Flatten()( l2 ); //(7*7*64)
    auto l4 = ceras::Dropout(0.25f)( ceras::LeakyReLU(0.2f)( ceras::Dense( 64, 7*7*64 )( l3 ) ) );
    auto output = ceras::sigmoid( ceras::Dense( 1, 64 )( l4 ) );
    return ceras::model{ input, output };
}

int main()
{
    ceras::random_generator.seed( 42 );

    unsigned long const latent_dim = 16;
    unsigned long const epochs = 1;
    unsigned long const batch_size = 600; // should work
    unsigned long const iterations = 60000 / batch_size;

    // build models
    auto discriminator = build_discriminator();
    auto generator = build_generator( latent_dim );
    auto z = ceras::Input(); // (latent_dim, )
    auto critic = discriminator( generator( z ) );
    auto combined = ceras::model{ z, critic };
    ceras::debug_log( "All models generated." );

    auto c = combined.compile( ceras::MeanAbsoluteError(), ceras::SGD( batch_size, 0.05f ) ); //
    auto d = discriminator.compile( ceras::MeanAbsoluteError(), ceras::SGD(batch_size, 0.05f ) ); //
    ceras::debug_log( "All models compiled." );

    // prepare data
    auto const& [x_training, y_training, x_test, y_test] = ceras::dataset::mnist::load_data();
    auto X = x_training.as_type<float>() / 127.5f - 1.0f;
    X.reshape( {60000, 28, 28, 1} );
    auto const& valid = ceras::ones<float>( {batch_size, 1} );
    auto const& fake = ceras::zeros<float>( {batch_size, 1} );
    ceras::debug_log( "All data prepared." );

#if 1
    ceras::tensor<float> cache{{batch_size, 28, 28, 1}};
    for ( auto e : ceras::range( epochs ) )
    {
        for ( auto idx : ceras::range( iterations ) )
        {
            //auto data = X.slice( batch_size*idx, batch_size*(idx+1) );
            auto const& noise = ceras::randn<float>( {batch_size, latent_dim} );
            auto const& gen_images = generator.predict( noise );

            d.trainable( true );
            auto loss_fake = d.train_on_batch( gen_images, fake );
            //auto loss_valid = d.train_on_batch( data, valid );
            std::copy_n( X.begin()+idx*batch_size*28*28*1, batch_size*28*28*1, cache.begin() );
            auto loss_valid = d.train_on_batch( cache, valid );

            d.trainable( false );
            auto loss_gan = c.train_on_batch( noise, valid );

            std::cout << "At epoch " << e << " iteration " << idx << ", the losses are " << loss_fake << " " << loss_valid << " " << loss_gan <<  "\r" << std::flush;
        }
        std::cout << std::endl;
    }

    generator.save_weights( "./tmp/dcgan" );
#else
    generator.load_weights( "./tmp/dcgan" );
#endif

    ceras::debug_log( "Trying to generate noises." );
    auto const& noise = ceras::randn<float>( {1, latent_dim} );

    ceras::debug_log( "Trying to make prediction." );
    auto gen_images = generator.predict( noise );
    gen_images += 1.0f;
    gen_images *= 127.5f;

    ceras::imageio::imwrite( std::string{"./tmp/dcgan_example.png"}, ceras::squeeze( gen_images ) );

    /*
    tensor<std::uint8_t> img{ {28, 28} };
    for ( auto idx : range( batch_size ) )
    {
        std::copy_n( gen_images.begin()+idx*batch_size, 28*28, img.begin() );
        imageio::imwrite( std::string{"./tmp/dcgan_"} + std::to_string(idx) + std::string{".png"}, img );
    }
    */

    return 0;
}

