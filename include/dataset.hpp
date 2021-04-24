#ifndef RKQSLRMXHSPFGGPQCNEPEBAKCXHNXQPMXETNTTXBWEWBIQHVCFRKRFSFMLXXXRYFUKHEXYIGL
#define RKQSLRMXHSPFGGPQCNEPEBAKCXHNXQPMXETNTTXBWEWBIQHVCFRKRFSFMLXXXRYFUKHEXYIGL

#include "./tensor.hpp"
#include "./includes.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/for_each.hpp"

namespace ceras::dataset
{

    namespace mnist
    {
        ///
        /// Loads the MNIST dataset.
        /// @param path Path where to cache the dataset locally. Default to "./dataset/mnist", should be updated if running the program somewhere else.
        /// @return A tuple of 4 tensors: x_train, y_train, x_test, y_test. x_train, x_test: uint8 arrays of grayscale image data with shapes (num_samples, 28, 28).
        ///  y_train, y_test: uint8 tensor of digit labels (integers in range 0-9) with shapes (num_samples, 10). Note: for digit 0, the corresponding array is `[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]`.
        ///
        /// Example usage:
        /// @code
        /// auto const& [x_train, y_train, x_test, y_test] = ceras::dataset::mnist::load_data("/home/feng/dataset/mnist");
        /// @endcode
        ///
        /// Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset, which is available under the terms of the Creative Commons Attribution-Share Alike 3.0 license.
        ///
        inline auto load_data( std::string const& path = std::string{"./dataset/mnist"} )
        {
            std::string const training_image_path = path + std::string{"/train-images-idx3-ubyte"};
            std::string const training_label_path = path + std::string{"/train-labels-idx1-ubyte"};
            std::string const test_image_path = path + std::string{"/t10k-images-idx3-ubyte"};
            std::string const test_label_path = path + std::string{"/t10k-labels-idx1-ubyte"};

			auto const& load_binary = []( std::string const& filename )
			{
				std::ifstream ifs( filename, std::ios::binary );
				better_assert( ifs.good(), "Failed to load data from ", filename );
				std::vector<char> buff{ ( std::istreambuf_iterator<char>( ifs ) ), ( std::istreambuf_iterator<char>() ) };
				std::vector<std::uint8_t> ans( buff.size() );
				std::copy( buff.begin(), buff.end(), reinterpret_cast<char*>( ans.data() ) );
				return ans;
			};

            auto const& extract_image = []( std::vector<std::uint8_t> const& image_data )
            {
                unsigned long const offset = 16;
                unsigned long const samples = (image_data.size()-offset) / (28*28);
                tensor<std::uint8_t> ans{ {samples, 28, 28} };
                std::copy( image_data.begin()+offset, image_data.end(), ans.data() );
                return ans;
            };

            auto const& extract_label = []( std::vector<std::uint8_t> const& label_data )
            {
                unsigned long const offset = 8;
                unsigned long const samples = label_data.size() - offset;
                auto ans = zeros<std::uint8_t>({samples, 10});
                auto ans_2d = matrix{ ans.data(), samples, 10 };
                for ( auto idx : range( samples ) )
                    ans_2d[idx][label_data[idx+offset]] = 1;
                return ans;
            };

            return std::make_tuple( extract_image(load_binary(training_image_path)),
                                    extract_label(load_binary(training_label_path)),
                                    extract_image(load_binary(test_image_path)),
                                    extract_label(load_binary(test_label_path)) );
        }
    }

#if 0
    namespace cifar10
    {
        inline auto load_data( std::string const& path = std::string{} )
        {
        }
    }

    namespace cifar100
    {
        inline auto load_data( std::string const& path = std::string{} )
        {
        }
    }

    namespace imdb
    {
        inline auto load_data( std::string const& path = std::string{} )
        {
        }
    }

    namespace reuters
    {
        inline auto load_data( std::string const& path = std::string{} )
        {
        }
    }

    namespace fashion_mnist
    {
        inline auto load_data( std::string const& path = std::string{} )
        {
        }
    }

    namespace boston_housing
    {
        inline auto load_data( std::string const& path = std::string{} )
        {
        }
    }
#endif


}//namespace ceras

#endif//RKQSLRMXHSPFGGPQCNEPEBAKCXHNXQPMXETNTTXBWEWBIQHVCFRKRFSFMLXXXRYFUKHEXYIGL

