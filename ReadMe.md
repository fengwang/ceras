# Ceras

----


__ceras__ is yet another tiny deep learning engine.  This library mimiks tensorflow 1.x APIs, in pure C++20 and is header-only. CUDA acceleration is optional to _convolutional_ and _dense_ layers, as __ceras__ is written for ordinary devices such as a gaming laptop with a GeForce GTX 1060, in which the GPU memory is limited.

----


## Table of contents

* [Features](#features)
* [Build](#build)
* [Design](#design)
* [Examples](#examples)
* [Supported layers/operations](#supported-layers)
* [TODO](#todo)
* [License](#license)
* [Acknowledgements](#acknowledgements)


----

## Features
- Fast, with/without GPU:
    - 98% accuracy on MNIST in 10 epochs in 30s (loading dataset, training and validation on a laptop with Intel(R) Core(TM) i7-7700HQ and a mobile GTX 1060)
- Portable:
    - compiles anywhere as long as your compiler supports C++20;
    - CUDA acceleration is optional, not a must;
    - header-only.
- Simply implemented:
    - mimicking Tensorflow grammar, but in C++.
    - minimizing the levels of indirection to expose as many implementation details as possible.


## Build
__Using this library__:

copy the `include` directory to the working directory, then include the header file

```cpp
#include "ceras.hpp"
```

**Compile/link**:

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

## Design



### [tensor](./include/tensor.hpp)
A `tensor` variable holds a multiple dimensional array.  A `tensor` instance can be generated by
```cpp
ceras::tensor<float> a{{2, 1, 2}, {0.0f, 0.1f, 0.2f, 0.3f}};
```
in which the template parameter `float` is for the data type, the first argument `{2, 1, 2}` is for the tensor shape, and the second argument `{0.0f, 0.1f, 0.2f, 0.3f}` is for the data stored in the tensor.

Quite a few operations, such as `+`, `-`, `*`,  `abs`,  `random`, `randn`, `reduce` and `max` are implemented for `tensor`. But these operations are there to serve the purpose of deep learning, not intend to be a generic tensor library.


### [constant](./include/constant.hpp)
A `constant` variable  holds a `tensor` instance, and this `tensor` is not supposed to be updated in its life-time.

```cpp
ceras::tensor<float> eye{{2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}};
ceras::constant<ceras::tensor<float>> c_eye{eye};
```

### [place_holder](./include/place_holder.hpp)
A `place_holder` variable holds a position that a `tensor` will be fed later.

```cpp
ceras::place_holder<ceras::tensor<float>> input{};
// ......
session<ceras::float<float>> s;
ceras::tensor<float> a{{2, 1, 2}, {0.0f, 0.1f, 0.2f, 0.3f}};
s.bind(input, a ); // binding a tensor to a place_holder
```

### [variable](./include/variable.hpp)

A `variable` variable  holds a stateful `tensor`, and this `tensor` will be updated anytime. This is designed for the weights in a neural network, which will be updated in every epoch of the training.

```cpp
auto w = ceras::variable{ ceras::randn<float>( {28*28, 256}, 0.0, 10.0/(28.0*16.0) ) };
```

### [operation](./include/operation.hpp) and  computation graph
__ceras__ uses [expression template](https://en.wikipedia.org/wiki/Expression_templates) to represent a computation graph. A computation graph is a directed graph in which each node corresponds to a `variable`, a `place_holder`, a `constant` or an `operation`.  In __ceras__, these node types are grouped in a `Expression` concept.

For example, a computation graph computes output _Expression_ `z` of two input _Expression_ `x` and `y`. Here `x` and `y` are two input nodes of `z`, and `z` is the consumer of `x` and `y`.

![x+y=z](./assets/x_y_z.png)

If `x` and `y` are two tensors are to be binded in a later stage, the corresponding code is
```cpp
auto x = ceras::place_holder<ceras::tensor<float>>{};
auto y = ceras::place_holder<ceras::tensor<float>>{};
auto z = x + y;
```

This kind of expression is more useful when the computation is getting more complex, for example `z = σ(A*x+b)`

![axb](./assets/axy.png)

in which `x`, `A` and `b` are `variable`s / `place_holder`s / `constant`s, and `*`, `+` and `σ` are `operations`s.

If `A` and `b` are two variables, and `x` is a place_holder, then the corresponding code is
```cpp
auto x = ceras::place_holder<ceras::tensor<float>>{};
auto A = ceras::variable{ ceras::ones<float>({3, 3}) };//just for demostration, should not be initialized to ones
auto b = ceras::variable{ ceras::zeros<float>({3,}) };
auto z = sigmoid( A*x + b );
```

### [session](./include/session.hpp)

To evaluate the operations (computation graph), we need a session.

```cpp
auto s = ceras::session<ceras::tensor<float>>{};
```

Then we can bind a tensor to `x`,
```cpp
auto X = ceras::tensor<float>{{3,}, {1.0f, 2.0f, 3.0f}};
s.bind(x, X);
```

And evaluate the output at node `z`:
```cpp
auto result = s.run(z);
```

This will generate a result tensor with shape `(3,)` and values `(0.997527, 0.997527,0.997527)`. In addition, the `x`,  `A` and `b` can also be evaluated by calling
```cpp
auto _x = s.run(x);
auto _A = s.run(A);
auto _b = s.run(b);
```

By design, an instance of an expression has a builtin `forward()` method. When a session runs an expression, the `forward()` method will be invoked.

Please find the complete code from [this file](./test/session.cc).

### [loss](./include/loss.hpp)

A `loss` variable provides a metric between the expected output and the actual output of the computation graph. And a `loss` is implemented as an `Expression`. For example, the `mae` loss can be defined as

```cpp
template < Expression Lhs_Expression, Expression Rhs_Expression >
auto constexpr mae( Lhs_Expression const& ground_truth, Rhs_Expression const& output ) noexcept
{
    return mean_reduce(abs(ground_truth - output));
};
```
in which `mean_reduce`, `abs` and `-` are predefined operations. Usually the `ground_truth` is just a place_holder variable, and will be rebinded at every training epoch.

We can define our loss operation with a place_holder for the ground_truth

```cpp
auto ground_truth = ceras::place_holder<tensor<float>>{};
auto loss = mae(ground_truth, z);
```

### [optimizer](./include/optimizer.hpp)

An `optimizer` variable holds an instance of an `expression` of loss. When an session runs an optimizer, the builtin method `forward()` will be invoked. And we define an optimizer this way:

```cpp
unsigned long batch_size = ...;
float learning_rate = ...;
auto optimizer = ceras::sgd{loss, batch_size, learning_rate};
```

In a single epoch, we first execute a forward pass on the loss, with input `x` and `ground_truth` having been binded:
```cpp
s.bind( x, ...);
s.bind(ground_truth, ...);
s.run(loss);
```

then we execute a backward pass with the optimizer:
```cpp
s.run(optimizer);
```

By repeating forward pass and backward pass multiple times, the weights A and the bias b can be gradually updated.


### more details

__TODO__

## Examples


### implement VGG16

There are a few pre-defined layers in file `./include/layer.hpp`, such as `Input`, `Conv2D` and `Dense`. Starting from thses layers, we are already able to build a VGG16 model.


The input layer for VGG16 is defined as
```cpp
auto input = Input(); //  3D tensor input, (batch_size, 224, 224, 3)
```
followed by a convolutional layer and a relu activation
```cpp
auto l0 = relu( Conv2D( 64, {3, 3}, {224, 3, 3}, "same" )(input) ); // 224, 224, 64
```
and a max pooling layer
```cpp
auto l1 = max_pooling_2d( 2 ) ( l0 ); // 112, 112, 64
```
Then 2 convolutional layers and a max pooling layer
```cpp
auto l2 = relu( Conv2D( 128, {3, 3}, {112, 112, 64}, "same" )( l1 ) ); // 112, 112, 128
auto l3 = relu( Conv2D( 128, {3, 3}, {112, 112, 128}, "same" )( l2 ) ); // 112, 112, 128
auto l4 = max_pooling_2d( 2 ) ( l3 ); // 56, 56, 128
```
followed by 3 convolutional layers and a max pooling layer
```cpp
auto l5 = relu( Conv2D( 256, {3, 3}, {56, 56, 128}, "same" )( l4 ) ); // 56, 56, 256
auto l6 = relu( Conv2D( 256, {3, 3}, {56, 56, 256}, "same" )( l5 ) ); // 56, 56, 256
auto l7 = relu( Conv2D( 256, {3, 3}, {56, 56, 256}, "same" )( l6 ) ); // 56, 56, 256
auto l8 = max_pooling_2d( 2 ) ( l7 ); // 28, 28, 256
```
followed by another 3 convolutional layers and a max pooling layer
```cpp
auto l9 = relu( Conv2D( 512, {3, 3}, {28, 28, 256}, "same" )( l8 ) ); // 28, 28, 512
auto l10 = relu( Conv2D( 512, {3, 3}, {28, 28, 512}, "same" )( l9 ) ); // 28, 28, 512
auto l11 = relu( Conv2D( 512, {3, 3}, {28, 28, 512}, "same" )( l10 ) ); // 28, 28, 512
auto l12 = max_pooling_2d( 2 ) ( l11 ); // 14, 14, 512
```
and again
```cpp
auto l13 = relu( Conv2D( 512, {3, 3}, {14, 14, 512}, "same" )( l12 ) ); // 14, 14, 512
auto l14 = relu( Conv2D( 512, {3, 3}, {14, 14, 512}, "same" )( l13 ) ); // 14, 14, 512
auto l15 = relu( Conv2D( 512, {3, 3}, {14, 14, 512}, "same" )( l14 ) ); // 14, 14, 512
auto l16 = max_pooling_2d( 2 ) ( l15 ); // 7, 7, 512
```
then this 3d layer is flattened to 1d
```cpp
auto l17 = flatten( l16 ); // 7x7x512
```
followed by a dense layer
```cpp
auto l18 = relu( Dense( 4096, 7*7*512 )( l17 ) ); // 4096
```
and then 2 dense layers to the output layer
```cpp
auto l19 = relu( Dense( 4096, 4096 )( l18 ) ); // 4096
auto l20 = relu( Dense( 1000, 4096 )( l19 ) ); // 1000
auto output = l20;
```

With above codes, VGG16 model has been build. However, we not train this model here as we do not have the training set yet. But we can demonstrate the training process with mnist, which is a dataset much smaller than imagenet.



### [defining a 3-layered NN, 256+128 hidden units](./test/mnist_mini.cc) for mnist

**define a 3 layer model**

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

Note: this convolutional model uses `drop_out`, when training this model, we should set `ceras::learning_phase = 1;`, which is the default value; and when doing prediction using this model, we should set `ceras::learning_phase = 0;`. This is also the case for `BatchNormalization`. The reason is that, the forward propagation behaviours for `drop_out` and `BatchNormalization` layers are different between the training and the prediction phase.


## Supported layers
+ [Operations](./include/operation.hpp):
    - [`plus`](#plus), or operator `+`;
    - [`multiply`](#multiply), or operator `*`, note this operation implies matrix-matrix multiplication, i.e., `dot` in numpy;
    - [`log`](#log);
    - `negative`;
    - `elementwise_product`, or `hadamard_product`;
    - `sum_reduct`;
    - `mean_reduce`;
    - `minus`;
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
    - `concatenate`, or `concat`;
    - `maximum`.
+ [Activations](./include/activation.hpp):
    - [`softmax`](#softmax);
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
+ [Losses](./include/loss.hpp):
    - [`mae`](#mae);
    - `mse`;
    - `cross_entropy`;
    - `hinge_loss`.
+ [Optimizers](./include/optimizer.hpp):
    - `sgd`;
    - `adagrad`;
    - `rmsprop`;
    - `adadelta`;
    - `adam`;
    - [`gradient_descent`](#gradient_descent).

### plus

`plus` or `+` does element-wise addition. (note broadcasting is permitted.)

```cpp
    auto a = variable{ ones<float>( {2, 2} ) };
    auto b = variable{ zeros<float>( {2, 2} ) };
    auto ab = a+b; // or 'auto ab = plus( a, b );'
    ceras::session<ceras::tensor<double>> s;
    std::cout <<  s.run( ab );
```
this will produce a 2x2 matrix of `[ [1, 1], [1, 1] ]`. Full code is [here](./test/layer_plus.cc).

### multiply

`multiply` or `*` does matrix multiplication.

```cpp
    auto a = variable{ ones<float>( {2, 2} ) };
    auto b = variable{ ones<float>( {2, 2} ) };
    auto ab = a*b; // or 'auto ab = multiply( a, b );'
    ceras::session<ceras::tensor<double>> s;
    std::cout <<  s.run( ab );
```
this will produce a 2x2 matrix of [[2, 2], [2, 2]]. Full code is [here](./test/layer_multiply.cc).

### log

`log` does element-wise logarithm on each element.

```cpp
    auto a = variable{ ones<float>( {2, 2} ) };
    auto la = log(a);
    ceras::session<ceras::tensor<double>> s;
    std::cout <<  s.run( la );
```

this will produce a 2x2 matrix of [[0, 0], [0, ]]. Full code is [here](./test/layer_log.cc).

### softmax

`softmax` applies softmax on last channel elements.

```cpp
    auto a = variable{ ones<float>( {2, 2} ) };
    auto ls = softmax(a);
    ceras::session<ceras::tensor<double>> s;
    std::cout <<  s.run( ls );
```

this will produce a 2x2 matrix of [[0.5, 0.5], [0.5, 0.5]]. Full code is [here](./test/layer_softmax.cc).

### mae

`mae` gives out mean absolute error.

```cpp
    auto a = variable{ ones<float>( {2, 2} ) };
    auto b = variable{ zeros<float>( {2, 2} ) };
    auto ab = mae(a, b);
    ceras::session<ceras::tensor<double>> s;
    std::cout <<  s.run( ab );
```

this will produce a 1x1 matrix of [1]. Full code is [here](./test/layer_mae.cc).

### gradient_descent


`gradient_decent` is an optimizer taking 3 arguments:

- a loss expression
- a batch_size
- a learning rate

A typical optimizer instance is `auto optimizer = gradient_decent{ loss, batch_size, learning_rate };`

```cpp
    // define model, a single layer NN, using softmax activation
    auto x = place_holder<tensor<double>>{};
    auto W = variable{ tensor<double>{ {2, 2}, {1.0, -1.0, 1.0, -1.0} } };
    auto b = variable{ tensor<double>{{1,2}, {0.0, 0.0} } };
    auto p = softmax( x * W + b ); // p is our model

    // preparing input for the model
    unsigned long const N = 512;
    auto blues = randn<double>( {N, 2} ) - 2.0 * ones<double>( {N, 2} );
    auto reds = randn<double>( {N, 2} ) + 2.0 * ones<double>( {N, 2} );
    auto _x = concatenate( blues, reds, 0 );

    // binding input to layer x
    session<tensor<double>> s;
    s.bind( x, _x );

    // define loss here
    auto c = place_holder<tensor<double>>{};
    auto J = cross_entropy( c, p );

    // generating output/ground_truth for the model
    auto c_blue = tensor<double>{{1, 2}, {1.0, 0.0} };
    auto c_blues = repmat( c_blue, N, 1 );
    auto c_red = tensor<double>{{1, 2}, {0.0, 1.0} };
    auto c_reds = repmat( c_red, N, 1 );
    auto _c = concatenate( c_blues, c_reds, 0 );

    // binding output to the model
    s.bind( c, _c );
    // define optimizer here
    double const learning_rate = 1.0e-3;
    auto optimizer = gradient_descent{ J, 1, learning_rate }; // J is the loss, 1 is the batch size, learning_rate is the hyper-parameter

    auto const iterations = 32UL;
    for ( auto idx = 0UL; idx != iterations; ++idx )
    {
        // first do forward propagation
        auto J_result = s.run( J );
        std::cout << "J at iteration " << idx+1 << ": " << J_result[0] << std::endl;
        // then do backward propagation
        s.run( optimizer );
    }
```

Fixing the random seed to 42 by `random_generator.seed( 42 );`, we can get output below:

```
J at iteration 1: 8165.29
J at iteration 2: 633.804
J at iteration 3: 9.5146
J at iteration 4: 3.41902
J at iteration 5: 2.33691
J at iteration 6: 1.8801
J at iteration 7: 1.6095
J at iteration 8: 1.41938
J at iteration 9: 1.27353
J at iteration 10: 1.1565
J at iteration 11: 1.06044
J at iteration 12: 0.980601
J at iteration 13: 0.913777
J at iteration 14: 0.857562
J at iteration 15: 0.810077
J at iteration 16: 0.769802
J at iteration 17: 0.735497
J at iteration 18: 0.706139
J at iteration 19: 0.680886
J at iteration 20: 0.65904
J at iteration 21: 0.640021
J at iteration 22: 0.623354
J at iteration 23: 0.608642
J at iteration 24: 0.595558
J at iteration 25: 0.58383
J at iteration 26: 0.573235
J at iteration 27: 0.563587
J at iteration 28: 0.554732
J at iteration 29: 0.546542
J at iteration 30: 0.538911
J at iteration 31: 0.531752
J at iteration 32: 0.524992
```

The full code is [here](./test/optimize.cc).


## TODO
+ save/load model
+ mimicking Tensorflow::Keras grammar
+ recurrent operations
+ provide a single-header file


## License

+ BSD


## Acknowledgements

+ [Tensorflow 1](https://www.tensorflow.org/)
+ [TensorSlow](https://github.com/danielsabinasz/TensorSlow)
+ [Caffe](https://github.com/BVLC/caffe)
+ [stb](https://github.com/nothings/stb)
+ [glob](https://github.com/p-ranav/glob)
+ [tqdm-cpp](https://github.com/mraggi/tqdm-cpp)

