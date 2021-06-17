#include "../include/ceras.hpp"
#include "../include/utils/imageio.hpp"
#include <iostream>

#if 0
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
#endif
//using namespace ceras;
auto build_generator( unsigned long const latent_dim )
{
#if 0
    using namespace ceras;
    auto input = Input();
    auto l0 = ReLU( Dense( 128*7*7, latent_dim )( input ) );
    auto l1 = Reshape({7, 7, 128})( l0 );
    auto l2 = UpSampling2D( 2 )( l1 );
    auto l3 = Conv2D( 128, {3, 3}, {14, 14, 128}, "same" )( l2 );
    auto l4 = BatchNormalization({14, 14, 128}, 0.8f)( l3 );
    auto l5 = ReLU( l4 );
    auto l6 = UpSampling2D( 2 )( l5 );
    auto l7 = Conv2D( 64, {3, 3}, {28, 28, 128}, "same" )( l6 );
    auto l8 = BatchNormalization({28, 28, 64}, 0.8f)( l7 );
    auto l9 = ReLU( l8 );
    auto l10 = Conv2D( 1, {3, 3}, { 28, 28, 64 }, "same" )( l9 );
    auto l11 = tanh( l10 );
    return model{ input, l11 };

#else
    auto input = ceras::Input(); // (latent_dim, )
    auto l0 = ceras::ReLU( ceras::Dense( 1024, latent_dim )( input ) );
    auto l1 = ceras::ReLU( ceras::Dense( 1024, 1024 )( l0 ) );
    auto l2 = ceras::ReLU( ceras::Dense( 1024, 1024 )( l1 ) );
    auto l3 = ceras::tanh( ceras::Dense( 28*28, 1024 )( l1 ) );
    auto output = ceras::Reshape( {28, 28, 1} )( l3 );
    return ceras::model{ input, output };
#endif
}


#if 0
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
#endif
auto build_discriminator()
{
#if 0
    using namespace ceras;
    auto input = Input(); // (28, 28, 1)
    auto l0 = MaxPooling2D(2)(Conv2D( 32, {3, 3}, {28, 28, 1}, "same", {1, 1} )( input ));
    auto l1 = LeakyReLU( 0.2f )( l0 );
    auto l2 = Dropout( 0.25f )( l1 );
    auto l3 = MaxPooling2D( 2 )(Conv2D( 64, {3, 3}, {14, 14, 32}, "same", {1, 1})( l2 ));
    auto l4 = BatchNormalization( {7, 7, 64}, 0.8f )( l3 );
    auto l5 = LeakyReLU( 0.2f )( l4 );
    auto l6 = Dropout( 0.25f )( l5 );
    auto l7 = Flatten()( l6 );
    auto l8 = Dense( 1, 7*7*64 )( l7 );
    auto l9 = sigmoid( l8 );
    return model{ input, l9 };
#else
    auto input = ceras::Input(); // (28, 28, 1)
    auto l0 = ceras::Flatten()( input );
    auto l1 = ceras::ReLU( ceras::Dense( 1024, 28*28 )( l0 ) );
    auto l2 = ceras::Dropout(0.3f)(ceras::ReLU( ceras::Dense( 1024, 1024 )( l1 ) ));
    auto output = ceras::sigmoid( ceras::Dense( 1, 1024 )( l2 ) );
    return ceras::model{ input, output };
#endif
}

int main()
{
    ceras::random_generator.seed( 42 );

    unsigned long const latent_dim = 16;
    unsigned long const epochs = 50;
    unsigned long const batch_size = 60; // should work
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
    //auto c = combined.compile( ceras::MeanAbsoluteError(), ceras::Adam( batch_size, 0.01f, 0.5f ) ); //
    auto d = discriminator.compile( ceras::MeanAbsoluteError(), ceras::SGD(batch_size, 0.05f ) ); //
    //auto d = discriminator.compile( ceras::MeanAbsoluteError(), ceras::Adam(batch_size, 0.01f, 0.5f ) ); //
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

