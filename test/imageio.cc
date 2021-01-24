#include "../include/includes.hpp"

#include "../include/utils/imageio.hpp"

int main()
{
    // load png image
    int width, height, channels;
    unsigned char *img = stbi_load("./assets/asset_1.png", &width, &height, &channels, 0);

    if(img == nullptr)
        std::cout << "failed to load png file" << std::endl;
    else
        std::cout << "loaded png file with width: " << width << ", height: " << height << ", channels: " << channels << std::endl;

    // save jpg image
    stbi_write_jpg("./assets/asset_1.jpg", width, height, channels, img, 100);
    stbi_write_png("./assets/asset_1_duplicated.png", width, height, channels, img,  width * channels );

    stbi_image_free(img);


    {
        using namespace ceras::imageio;
        auto image = imread( "./assets/asset_1.png" );
        if ( image.size() != 0 )
        {
            imwrite("./assets/asset_1_test.png", image);
            imwrite("./assets/asset_1_test.jpg", image);
            imwrite("./assets/asset_1_test.tga", image);
            imwrite("./assets/asset_1_test.bmp", image);
            imwrite("./assets/asset_1_test_no_extension", image);
        }
    }



    return 0;
}

