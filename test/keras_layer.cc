#include "../include/keras/layer.hpp"

int main()
{
    if (0)
    {
        using namespace ceras::keras;
        auto inp = Input().name("inp").shape( {32,} )();
        auto l1 = Dense().units( 128 ).name("dense1")( inp );
        auto l2 = Dense().units( 128 ).name("dense2")( l1 );

        auto output = l2;
        auto e = ceras::keras::construct_computation_graph( output );
        auto g = ceras::computation_graph( e );
        std::cout << g << std::endl;
    }

    if (1)
    {
        using namespace ceras::keras;
        auto inp = Input().name("inp").shape( {32,} )();
        auto l1 = Dense().units( 128 ).name("dense1").use_bias(false)( inp );
        auto l11 = ReLU().name( "relu_layer" )( l1 );
        auto l2 = Dense().units( 128 )( l11 );
        auto l3 = LeakyReLU().name("leaky_reul").alpha(0.1f)( l2 );
        auto l4 = Dropout().rate(0.5f)( l3 );

        auto output = l4;
        auto e = ceras::keras::construct_computation_graph( output );
        auto g = ceras::computation_graph( e );
        std::cout << g << std::endl;
    }



#if 0
    if (0)
    {
        auto inp = ceras::keras::Input( {32,} );
        auto l1 = ceras::keras::Dense( 128 )( inp );
        auto l2 = ceras::keras::Dense( 128 )( l1 );
        auto l3 = ceras::keras::ReLU( )( l2 );
        auto l4 = ceras::keras::Dense( 77 )( l3 );
        auto l5 = ceras::keras::Lisht( )( l4 );


        auto output = l5;
        auto e = ceras::keras::construct_computation_graph( output );
        auto g = ceras::computation_graph( e );
        std::cout << g << std::endl;
    }

    if (0)
    {
        auto inp = ceras::keras::Input( {128, 128, 1} );
        std::cout << fmt::format( "Got outptu shape {}", (*(std::get<0>(inp))).compute_output_shape() ) << std::endl;
        auto l1 = ceras::keras::Conv2D( 32, {3, 3}, "same" )( inp );
        std::cout << fmt::format( "Got outptu shape {}", (*(std::get<0>(l1))).compute_output_shape() ) << std::endl;
        auto l2 = ceras::keras::Conv2D( 32, {3, 3}, "valid" )( l1 );
        std::cout << fmt::format( "Got outptu shape {}", (*(std::get<0>(l2))).compute_output_shape() ) << std::endl;
        auto l3 = ceras::keras::ReLU()( l2 );
        std::cout << fmt::format( "Got outptu shape {}", (*(std::get<0>(l3))).compute_output_shape() ) << std::endl;
        auto l4 = ceras::keras::BatchNormalization(0.9f)( l3 );
        std::cout << fmt::format( "Got outptu shape {}", (*(std::get<0>(l4))).compute_output_shape() ) << std::endl;


        auto output = l4;
        auto e = ceras::keras::construct_computation_graph( output );
        auto g = ceras::computation_graph( e );
        std::cout << g << std::endl;
    }

    if (0)
    {
        auto inp = ceras::keras::Input( {128, 128, 1} );
        std::cout << fmt::format( "Got outptu shape {}", (*(std::get<0>(inp))).compute_output_shape() ) << std::endl;
        auto l1 = ceras::keras::Concatenate()( inp, inp );
        std::cout << fmt::format( "Got outptu shape {}", (*(std::get<0>(l1))).compute_output_shape() ) << std::endl;

        auto output = l1;
        auto e = ceras::keras::construct_computation_graph( output );
        auto g = ceras::computation_graph( e );
        std::cout << g << std::endl;
    }

    {
        auto inp = ceras::keras::Input( {128, 128, 1} );
        std::cout << fmt::format( "Got outptu shape {}", (*(std::get<0>(inp))).compute_output_shape() ) << std::endl;
        auto l1 = ceras::keras::Reshape({16, 8, 128 })( inp );
        std::cout << fmt::format( "Got outptu shape {}", (*(std::get<0>(l1))).compute_output_shape() ) << std::endl;
        auto l2 = ceras::keras::MaxPooling2D( 2 )( l1 );
        std::cout << fmt::format( "Got outptu shape {}", (*(std::get<0>(l2))).compute_output_shape() ) << std::endl;
        auto l3 = ceras::keras::UpSampling2D( 2 )( l2 );
        std::cout << fmt::format( "Got outptu shape {}", (*(std::get<0>(l3))).compute_output_shape() ) << std::endl;

        auto output = l3;
        auto e = ceras::keras::construct_computation_graph( output );
        auto g = ceras::computation_graph( e );
        std::cout << g << std::endl;
    }
#endif

    return 0;
}

