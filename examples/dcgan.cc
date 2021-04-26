#include "../include/ceras.hpp"
#include <iostream>

using namespace ceras;

auto build_generator( unsigned long const latent_dim )
{
    auto input = Input(); // (latent_dim, )
    auto l1 = ReLU( Dense( 128*7*7, latent_dim )( input ) );
    auto l2 = Reshape( {7, 7, 128} )( l1 );
    auto l3 = UpSampling2D( 2 )( l2 ); // (14, 14, 128)
    auto l4 = Conv2D( 128, {3, 3}, {14, 14, 128}, "same" )( l3 );
    //auto l5 = ReLU( BatchNormalization( 0.8f, {14, 14, 128} )( l4 ) );
    auto l5 = ReLU( l4 );
    auto l6 = UpSampling2D( 2 )( l5 ); // (28, 28, 128)
    auto l7 = Conv2D( 64, {3, 3}, {28, 28, 128}, "same" )( l6 );
    //auto l8 = ReLU( BatchNormalization( 0.8f, {28, 28, 64} )( l7 ) );
    auto l8 = ReLU( l7 );
    auto output = tanh( Conv2D( 1, {3,3}, {28, 28, 64}, "same" )( l8 ) );
    return model{ input, output };
}

auto build_discriminator()
{
    auto input = Input(); // (28, 28, 1)
    auto l1 = MaxPooling2D(2)( Dropout(0.25f)( LeakyReLU(0.2f)( Conv2D( 32, {3, 3}, {28, 28, 1}, "same" )( input ) ) ) ); // (14, 14, 32)
    //auto l2 = MaxPooling2D(2)( Dropout(0.25f)( LeakyReLU(0.2f)( BatchNormalization(0.8f, {14, 14, 64})( LeakyReLU(0.2f)( Conv2D( 64, {3, 3}, {14, 14, 32}, "same" )( l1 ) ) ) ) ) ); //(7, 7, 64)
    auto l2 = MaxPooling2D(2)( Dropout(0.25f)( LeakyReLU(0.2f)( LeakyReLU(0.2f)( Conv2D( 64, {3, 3}, {14, 14, 32}, "same" )( l1 ) ) ) ) ); //(7, 7, 64)
    auto l3 = Flatten()( l2 ); //(7*7*64)
    auto l4 = Dropout(0.25f)( LeakyReLU(0.2f)( Dense( 64, 7*7*64 )( l3 ) ) );
    auto output = sigmoid( Dense( 1, 64 )( l4 ) );
    return model{ input, output };
}

int main()
{
    random_generator.seed( 42 );

    unsigned long const latent_dim = 16;
    unsigned long const epochs = 1;
    unsigned long const batch_size = 600; // should work
    unsigned long const iterations = 60000 / batch_size;

    // build models
    auto discriminator = build_discriminator();
    auto generator = build_generator( latent_dim );
    auto z = Input(); // (latent_dim, )
    auto critic = discriminator( generator( z ) );
    auto combined = model{ z, critic };
    debug_log( "All models generated." );

    //auto c = combined.compile( CategoricalCrossentropy(), SGD( batch_size, 0.05f ) ); //
    //auto d = discriminator.compile( CategoricalCrossentropy(), SGD(batch_size, 0.05f ) ); //
    auto c = combined.compile( MeanAbsoluteError(), SGD( batch_size, 0.05f ) ); //
    auto d = discriminator.compile( MeanAbsoluteError(), SGD(batch_size, 0.05f ) ); //
    debug_log( "All models compiled." );

    // prepare data
    auto const& [x_training, y_training, x_test, y_test] = dataset::mnist::load_data();
    auto X = x_training.as_type<float>() / 127.5f - 1.0f;
    X.reshape( {60000, 28, 28, 1} );
    auto const& valid = ones<float>( {batch_size, 1} );
    auto const& fake = zeros<float>( {batch_size, 1} );
    debug_log( "All data prepared." );

    for ( auto e : range( epochs ) )
    {
        for ( auto idx : range( iterations ) )
        {
            auto data = X.slice( batch_size*idx, batch_size*(idx+1) );
            auto const& noise = randn<float>( {batch_size, latent_dim} );
            auto const& gen_images = generator.predict( noise );

            d.trainable( true );
            auto loss_fake = d.train_on_batch( gen_images, fake );
            auto loss_valid = d.train_on_batch( data, valid );

            d.trainable( false );
            auto loss_gan = c.train_on_batch( noise, valid );

            std::cout << "At epoch " << e << " iteration " << idx << ", the losses are " << loss_fake << " " << loss_valid << " " << loss_gan <<  "\r" << std::flush;
        }
        std::cout << std::endl;
    }

    return 0;
}

