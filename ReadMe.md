# CERAS: yet another tiny deep learning engine

----

## Example Usage:

**using this library**:

copy the `include` directory to your working directory, then in your source code

```cpp
#include "ceras.hpp"
```

**compiliation/link**:

```bash
clang++ -c -std=c++2a -Wall -Wextra -ferror-limit=1 -ftemplate-backtrace-limit=0 -funsafe-math-optimizations  -Ofast -flto -pipe -march=native -DDEBUG -o ./obj/test_mnist.o test/mnist.cc
clang++ -o ./bin/test_mnist ./obj/test_mnist.o -funsafe-math-optimizations  -Ofast -flto -pipe -march=native
```

Enabling CUDA by defining macro `CUDA`: (tested with cuda 11.2.r11.2, gcc 10.2.0)

```bash
g++ -c -std=c++2a -Wall -Wextra -fmax-errors=1 -ftemplate-backtrace-limit=0 -funsafe-math-optimizations  -Ofast -flto -pipe -march=native -DCUDA -o ./obj/test_mnist.o test/mnist.cc
g++ -funsafe-math-optimizations  -Ofast -flto -pipe -march=native -o ./bin/test_mnist ./obj/test_mnist.o -L/opt/cuda/lib64 -pthread  -lcudart -lcublas
```

## mnist model

### [defining a 3-layered NN, 256+128 hidden units](./test/mnist.cc)

**define the model**

```cpp
// define computation graph, a 3-layered dense net with topology 784x256x128x10
using namespace ceras;
auto input = place_holder<tensor<float>>{}; // 1-D, 28x28 pixels

// 1st layer
auto w1 = variable{ randn<float>( {28*28, 256}, 0.0, 10.0/(28.0*16.0) ) };
auto b1 = variable{ zeros<float>( { 1, 256 } ) };

auto l1 = relu( input * w1 + b1 );
/* alternatively with BN
auto l1_1 = input * w1 + b1;
auto gamma = variable{ ones<float>( {1, 256} ) };
auto beta = variable{ zeros<float>( {1, 256} ) };
auto l1 = relu( batch_normalization(0.95)( l1_1, gamma, beta ) );
*/

// 2nd layer
auto w2 = variable{ randn<float>( {256, 128}, 0.0, 3.14/(16.0*11.2 )) };
auto b2 = variable{ zeros<float>( { 1, 128 } ) };
//auto l2 = relu( l1 * w2 + b2 );
auto l2 = sigmoid( l1 * w2 + b2 );

// 3rd layer
auto w3 = variable{ randn<float>( {128, 10}, 0.0, 1.0/35.8 ) };
auto b3 = variable{ zeros<float>( { 1, 10 } ) };
auto output = l2 * w3 + b3;

auto ground_truth = place_holder<tensor<float>>{}; // 1-D, 10
auto loss = cross_entropy_loss( ground_truth, output );
```

**preparing dataset**

```cpp
std::size_t const batch_size = 10;
tensor<float> input_images{ {batch_size, 28*28} };
tensor<float> output_labels{ {batch_size, 10} };

std::size_t const epoch = 1;
std::size_t const iteration_per_epoch = 60000/batch_size;
```

**prepare session**

```cpp
// creating session
session<tensor<float>> s;
s.bind( input, input_images );
s.bind( ground_truth, output_labels );
```

**define optimizer**

```cpp
float learning_rate = 1.0e-1f;
auto optimizer = gradient_descent{ loss, batch_size, learning_rate };
```


**start training**

```cpp
for ( auto e : range( epoch ) )
{
    for ( auto i : range( iteration_per_epoch ) )
    {
        s.run( loss ); //forward pass
        s.run( optimizer ); //backward pass
    }
}
```

**make prediction**

```cpp
std::size_t new_batch_size = 1;
tensor<float> new_input_images{ {new_batch_size, 28 * 28} };
s.bind( input, new_input_images );

for ( auto i : range( tests ) )
{
    //prepare new_input_images as inputs
    auto precition = s.run( output );
    //post precess prediction
}
```

### [alternative] [define a convolutional model](./test/mnist_conv2d.cc)

```cpp
using namespace ceras;
auto input = place_holder<tensor<float>>{}; // 1-D, 28x28 pixels

auto l0 = reshape( {28, 28, 1} )( input );

auto k1 = variable{ randn<float>( {32, 3, 3, 1}, 0.0, 10.0/std::sqrt(32.0*3*3*1) ) };
auto l1 = relu( conv2d(28, 28, 1, 1, 1, 1, "valid" )( l0, k1 ) ); // 26, 26, 32

auto l2 = max_pooling_2d( 2 ) ( l1 ); // 13, 13, 32

auto k2 = variable{ randn<float>( {64, 3, 3, 32}, 0.0, 10.0/std::sqrt(64.0*3*3*1) ) };
auto l3 = relu( conv2d(13, 13, 1, 1, 1, 1, "valid")( l2, k2 ) ); // 11, 11, 64

auto l4 = max_pooling_2d( 2 )( l3 ); //5, 5, 64
auto l5 = drop_out(0.5)( flatten( l4 ) );

auto w6 = variable{ randn<float>( {5*5*64, 10}, 0.0, 10.0/std::sqrt(7.0*7*64*10) ) };
auto b6 = variable{ zeros<float>( {1, 10} ) };

auto l6 = l5 * w6 + b6;
auto output = l6;

auto ground_truth = place_holder<tensor<float>>{}; // 1-D, 10
auto loss = cross_entropy_loss( ground_truth, output );
```

## License

+ AGPLv3
+ Anti-996


## Acknowledgements

+ [Tensorflow 1](https://www.tensorflow.org/)
+ [TensorSlow](https://github.com/danielsabinasz/TensorSlow)
+ [Caffe](https://github.com/BVLC/caffe)

