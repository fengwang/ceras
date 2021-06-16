#include "../include/ceras.hpp"
#include "../include/utils/imageio.hpp"
#include <iostream>

//using namespace ceras;
auto build_generator( unsigned long const latent_dim )
{
    auto input = ceras::Input(); // (latent_dim, )
    auto l0 = ceras::ReLU( ceras::Dense( 1024, latent_dim )( input ) );
    auto l1 = ceras::ReLU( ceras::Dense( 1024, 1024 )( l0 ) );
    auto l2 = ceras::ReLU( ceras::Dense( 1024, 1024 )( l1 ) );
    auto l3 = ceras::tanh( ceras::Dense( 28*28, 1024 )( l1 ) );
    auto output = ceras::Reshape( {28, 28, 1} )( l3 );
    return ceras::model{ input, output };
}

auto build_discriminator()
{
    auto input = ceras::Input(); // (28, 28, 1)
    auto l0 = ceras::Flatten()( input );
    auto l1 = ceras::ReLU( ceras::Dense( 1024, 28*28 )( l0 ) );
    auto l2 = ceras::ReLU( ceras::Dense( 1024, 1024 )( l1 ) );
    auto output = ceras::sigmoid( ceras::Dense( 1, 1024 )( l2 ) );
    return ceras::model{ input, output };
}

int main()
{
    ceras::random_generator.seed( 42 );

    unsigned long const latent_dim = 16;
    unsigned long const epochs = 20;
    unsigned long const batch_size = 10; // should work
    unsigned long const iterations = 60000 / batch_size;

    // build models
    auto discriminator = build_discriminator();

    auto generator = build_generator( latent_dim );

    auto z = ceras::Input(); // (latent_dim, )
    auto dis = discriminator;
    dis.trainable(false);
    auto critic = dis( generator( z ) );
    auto combined = ceras::model{ z, critic };

    combined.summary( "./dcgan_combined.dot" );
    discriminator.summary( "./dcgan_discriminator.dot" );
    generator.summary( "./dcgan_generator.dot" );

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

    ceras::tensor<float> cache{{batch_size, 28, 28, 1}};
    for ( auto e : ceras::range( epochs ) )
    {
        for ( auto idx : ceras::range( iterations ) )
        {
            auto const& noise = ceras::randn<float>( {batch_size, latent_dim} );
            auto const& gen_images = generator.predict( noise );

            //d.trainable( true );
            auto loss_fake = d.train_on_batch( gen_images, fake );
            std::copy_n( X.begin()+idx*batch_size*28*28*1, batch_size*28*28*1, cache.begin() );
            auto loss_valid = d.train_on_batch( cache, valid );

            //d.trainable( false );
            auto loss_gan = c.train_on_batch( noise, valid );

            std::cout << "At epoch " << e << " iteration " << idx << ", the losses are " << loss_fake << " " << loss_valid << " " << loss_gan <<  "\r" << std::flush;
        }
        std::cout << std::endl;

        auto const& noise = ceras::randn<float>( {1, latent_dim} );
        auto gen_images = generator.predict( noise );
        gen_images += 1.0f;
        gen_images *= 127.5f;
        ceras::imageio::imwrite( std::string{"./tmp/dcgan_example_"}+std::to_string(e)+std::string{".png"}, ceras::squeeze( gen_images ) );
    }

    generator.save_weights( "./tmp/dcgan" );

    ceras::debug_log( "Trying to generate noises." );
    auto const& noise = ceras::randn<float>( {1, latent_dim} );

    ceras::debug_log( "Trying to make prediction." );
    auto gen_images = generator.predict( noise );
    gen_images += 1.0f;
    gen_images *= 127.5f;

    ceras::imageio::imwrite( std::string{"./tmp/dcgan_example.png"}, ceras::squeeze( gen_images ) );

    return 0;
}

