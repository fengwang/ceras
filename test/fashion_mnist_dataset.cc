#include "../include/dataset.hpp"
#include "../include/utils/imageio.hpp"

#include <iostream>

int main()
{
    auto const&[training_images, training_labels, test_images, test_labels] = ceras::dataset::fashion_mnist::load_data();

    std::cout << "The training images shape is ";
    for ( auto r : training_images.shape() )
        std::cout <<  r << " ";
    std::cout << std::endl;

    // take a look at the first example
    ceras::tensor<std::uint8_t> snapshot{ {28, 28} };
    std::copy( training_images.begin(), training_images.begin()+28*28, snapshot.begin() );

    std::string path{ "./tmp/the_1st_fashion_mnist_image.png" };
    std::cout << "Saving the 1st sample to " << path << std::endl;
    ceras::imageio::imwrite( path, snapshot );
    std::cout << "Its label is [";
    for ( unsigned long idx = 0; idx != 10; ++idx )
        std::cout << static_cast<unsigned long>(training_labels[idx]) << " ";
    std::cout << "]" << std::endl;;

    return 0;
}

