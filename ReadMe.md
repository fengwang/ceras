<div align="center">
  <img src="https://github.com/fengwang/ceras/blob/main/assets/logo.png"><br><br>
</div>

----


__ceras__ is yet another tiny deep learning engine.  __ceras__ mimiks tensorflow 1.x APIs, in pure C++20 and header-only. CUDA acceleration is limited to _convolutional_ and _dense_ layers, as __ceras__ is written for ordinary devices such as a gaming laptop with a GeForce GTX 1060, in which the GPU memory is limited.

----


## Table of contents

* [Features](#features)
* [Build](#build)
* [Supported layers/operations](#supported-layers)
* [Concepts](#concepts)
* [Examples](#examples)
* [License](#license)
* [Acknowledgements](#acknowledgements)
* [TODO](#todo)


----

## Features
- Fast, with/without GPU:
    - 98% accuracy on MNIST in 10 epochs in 30s (loading dataset, training and validation on a laptop with Intel(R) Core(TM) i7-7700HQ and a mobile GTX 1060)
- Portable.
    - Runs anywhere as long as you have a compiler which supports C++20;
    - A Nvidia GPU is optional for acceleration, not a must;
    - header-only.
- Simply implemented:
    - mimicking TF grammar, but in C++.


## Build
**using this library**:

copy the `include` directory to the working directory, then inclue the header file

```cpp
#include "ceras.hpp"
```

**compiliation/link**:

```bash
g++ -c -std=c++20 -Wall -Wextra -ferror-limit=1 -ftemplate-backtrace-limit=0 -funsafe-math-optimizations  -Ofast -flto -pipe -march=native -DNDEBUG -o ./obj/test_mnist.o test/mnist.cc
g++ -o ./bin/test_mnist ./obj/test_mnist.o -funsafe-math-optimizations  -Ofast -flto -pipe -march=native
```

CUDA could be optionally enabled by defining macro `CUDA`: (tested with cuda 11.2.r11.2, gcc 10.2.0, note the compile/link options)

```bash
g++ -c -std=c++20 -Wall -Wextra -fmax-errors=1 -ftemplate-backtrace-limit=0 -funsafe-math-optimizations  -Ofast -flto -pipe -march=native -DCUDA -DNDEBUG -o ./obj/test_mnist.o test/mnist.cc
g++ -funsafe-math-optimizations  -Ofast -flto -pipe -march=native -o ./bin/test_mnist ./obj/test_mnist.o -L/opt/cuda/lib64 -pthread  -lcudart -lcublas
```

Note: As [Non-Type Template Parameters](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0732r2.pdf) is not yet implemented in clang, only gcc works with this library.

## Supported layers
+ Operations:
    - `plus`, or operator `+`;
    - `multiply`, or operator `*`, note this operation enables matrix-matrix multiplication, i.e., `dot` in numpy;
    - `log`;
    - `negative`;
    - `elementwise_product`, or `hadamard_product`;
    - `sum_reduct`;
    - `mean_reduce`;
    - `minus`, or operator `-`;
    - `square`;
    - `abs`;
    - `exp`;
    - `clip`;
    - `reshape`;
    - `flatten`;
    - `identity`;
    - `transpose`;
    - `conv2d`;
    - `drop_out`;
    - `max_pooling_2d`;
    - `average_pooling_2d`;
    - `up_sampling_2d`;
    - `batch_normalization`;
    - `instance_normalization`;
    - `concatenate`, or `concat`.
+ Activations:
    - `softmax`;
    - `selu`;
    - `softplus`;
    - `softsign`;
    - `sigmoid`;
    - `tanh`;
    - `relu`;
    - `leaky_relu`;
    - `elu`;
    - `exponential`;
    - `hard_sigmoid`;
    - `gelu`.
+ Losses:
    - `mae`;
    - `mse`;
    - `cross_entropy`.
+ Optimizers:
    - `sgd`;
    - `adagrad`;
    - `rmsprop`;
    - `adadelta`;
    - `adam`;
    - `gradient_descent`.

# Concepts

__TODO__


## Examples


### implement VGG16

First we define a convllution layer followed by a relu activation layer:

```cpp
using namespace ceras;
typedef tensor<float> tensor_type;

//
// example: Relu_Conv( 63, 3, {224, 224, 3} )( input_of_shape_224x224x3 );
//
inline auto Relu_Conv2D( unsigned long output_channels,std::vector<unsigned long> const& kernel_size, std::vector<unsigned long> const& input_shape )
{
    return [=]<Expression Ex>( Ex const& ex )
    {
        unsigned long const kernel_size_x = kernel_size[0];
        unsigned long const kernel_size_y = kernel_size[1];
        unsigned long const input_channels = input_shape[2];
        unsigned long const input_x = input_shape[0];
        unsigned long const input_y = input_shape[1];
        auto w = variable<tensor_type>{ glorot_uniform<float>({output_channels, kernel_size_x, kernel_size_y, input_channels}) };
        return relu( conv2d( input_x, input_y, 1, 1, 1, 1, "same" )( ex, w ) );
    };
}
```

Then we define a dense layer followed by a relu activation layer:

```cpp
//
// example: Relu_Dense( 512, 17 )( input_of_shape_17 );
//
inline auto Relu_Dense( unsigned long output_size, unsigned long input_size )
{
    return [=]<Expression Ex>( Ex const& ex )
    {
        auto w = variable<tensor_type>{ glorot_uniform<float>({input_size, output_size}) };
        auto b = variable<tensor_type>{ zeros<float>({1, output_size}) };
        return relu( ex * w + b );
    };
}
```

The input layer for VGG16 is defined as
```cpp
auto input = place_holder<tensor_type>{}; //  3D tensor input, (batch_size, 224, 224, 3)
```
followed by a convolutional layer
```cpp
auto l0 = Relu_Conv2D( 64, {3, 3}, {224, 224, 3} )( input ); // 224, 224, 64
```
and a max pooling layer
```cpp
auto l1 = max_pooling_2d( 2 ) ( l0 ); // 112, 112, 64
```
Then 2 convolutional layers and a max pooling layer
```cpp
auto l2 = Relu_Conv2D( 128, {3, 3}, {112, 112, 64} )( l1 ); // 112, 112, 128
auto l3 = Relu_Conv2D( 128, {3, 3}, {112, 112, 128} )( l2 ); // 112, 112, 128
auto l4 = max_pooling_2d( 2 ) ( l3 ); // 56, 56, 128
```
followed by 3 convolutional layers and a max pooling layer
```cpp
auto l5 = Relu_Conv2D( 256, {3, 3}, {56, 56, 128} )( l4 ); // 56, 56, 256
auto l6 = Relu_Conv2D( 256, {3, 3}, {56, 56, 256} )( l5 ); // 56, 56, 256
auto l7 = Relu_Conv2D( 256, {3, 3}, {56, 56, 256} )( l6 ); // 56, 56, 256
auto l8 = max_pooling_2d( 2 ) ( l7 ); // 28, 28, 256
```
followed by another 3 convolutional layers and a max pooling layer
```cpp
auto l9 = Relu_Conv2D( 512, {3, 3}, {28, 28, 256} )( l8 ); // 28, 28, 512
auto l10 = Relu_Conv2D( 512, {3, 3}, {28, 28, 512} )( l9 ); // 28, 28, 512
auto l11 = Relu_Conv2D( 512, {3, 3}, {28, 28, 512} )( l10 ); // 28, 28, 512
auto l12 = max_pooling_2d( 2 ) ( l11 ); // 14, 14, 512
```
and again
```cpp
auto l13 = Relu_Conv2D( 512, {3, 3}, {14, 14, 512} )( l12 ); // 14, 14, 512
auto l14 = Relu_Conv2D( 512, {3, 3}, {14, 14, 512} )( l13 ); // 14, 14, 512
auto l15 = Relu_Conv2D( 512, {3, 3}, {14, 14, 512} )( l14 ); // 14, 14, 512
auto l16 = max_pooling_2d( 2 ) ( l15 ); // 7, 7, 512
```
then this 3d layer is flattened to 1d
```cpp
auto l17 = flatten( l16 ); // 7x7x512
```
followed by a dense layer
```cpp
auto l18 = Relu_Dense( 4096, 7*7*512 )( l17 ); // 4096
```
and then 2 dense layers to the output layer
```cpp
auto l19 = Relu_Dense( 4096, 4096 )( l18 ); // 4096
auto l20 = Relu_Dense( 1000, 4096 )( l19 ); // 1000
auto output = l20;
```

Very similar to the behaviour of Tensorflow::Keras.



### [defining a 3-layered NN, 256+128 hidden units](./test/mnist_mini.cc) for mnist

**define the model**

```cpp
// define computation graph, a 3-layered dense net with topology 784x256x128x10
using namespace ceras;
auto input = Input();

// 1st layer
auto l1 = relu( Dense( 256, 28*28 )( input ) );
// or enabling BN
//auto l1 = relu( BatchNormalization( {256,} )( Dense( 256, 28*28 )( input ) ) );

// 2nd layer
auto l2 = sigmoid( Dense( 128, 256 )( l1 ) );

// 3rd layer
auto output = Dense( 10, 128 )( l2 );

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

### [alternative] [define a convolutional model](./test/mnist_conv2d_mini.cc)

```cpp
using namespace ceras;
auto input = Input(); // 28*28
auto l0 = reshape( {28, 28, 1} )( input ); // 28, 28, 1
auto l1 = relu( Conv2D( 32, {3, 3}, {28, 28, 1}, "valid" )( l0 ) );
auto l2 = max_pooling_2d( 2 ) ( l1 ); // 13, 13, 32
auto l3 = relu( Conv2D( 64, {3, 3}, {13, 13, 32}, "valid" )( l2 ) );
auto l4 = max_pooling_2d( 2 )( l3 ); //5, 5, 64
auto l5 = drop_out(0.5)( flatten( l4 ) );
auto output = Dense( 10, 5*5*64 )( l5 );

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

## TODO
+ save/load model
+ mimicking Tensorflow::Keras grammar
+ recurrent operations


