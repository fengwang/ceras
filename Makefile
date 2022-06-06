LOP           = -Wl,--gc-sections -flto
#OP            = -funsafe-math-optimizations  -Ofast -flto -pipe -march=native -DDEBUG -DCUDA
OP            = -funsafe-math-optimizations  -Ofast -flto -pipe -march=native -DDEBUG
OP            = -funsafe-math-optimizations  -Ofast -flto -pipe -march=native -DNDEBUG
OP            = -funsafe-math-optimizations  -Ofast -flto -pipe -march=native -DDEBUG -DCUDA
OP            = -pg -O0 -DDEBUG# -ggdb3
OP            = -funsafe-math-optimizations  -Ofast -flto -pipe -march=native -DNDEBUG
OP            = -funsafe-math-optimizations  -Ofast -flto -pipe -march=native -DNDEBUG -DCUDA
OP            = -funsafe-math-optimizations  -Ofast -flto -pipe -march=native -DNDEBUG -fsanitize=address
OP            = -funsafe-math-optimizations -fconcepts-diagnostics-depth=4  -Ofast -flto -pipe -march=native -DNDEBUG -DCUDA
OP            = -funsafe-math-optimizations -fconcepts-diagnostics-depth=4  -Ofast -flto -pipe -march=native -DNDEBUG
OP            = -funsafe-math-optimizations -fconcepts-diagnostics-depth=4  -Ofast -flto -pipe -march=native -DNDEBUG -DCUDA
OP            = -funsafe-math-optimizations -fconcepts-diagnostics-depth=4 -ftemplate-depth=999999 -Ofast -flto -pipe -march=native -DNDEBUG -DCUDA
#OP            = -funsafe-math-optimizations -ftemplate-depth=100860 -Ofast -ferror-limit=2 -flto -pipe -march=native -DDEBUG -DCUDA
OP            = -funsafe-math-optimizations -fconcepts-diagnostics-depth=4 -ftemplate-depth=100860 -Ofast -flto -pipe -march=native -DNDEBUG -fsanitize=address
OP            = -funsafe-math-optimizations -fconcepts-diagnostics-depth=4 -ftemplate-depth=100860 -Ofast -flto -pipe -march=native -DDEBUG
OP            = -funsafe-math-optimizations -fconcepts-diagnostics-depth=4 -ftemplate-depth=100860 -Ofast -flto -pipe -march=native -DDEBUG -DCBLAS
OP            = -fconcepts-diagnostics-depth=4  -O0 -pg -flto -pipe -march=native -DDEBUG
OP            = -funsafe-math-optimizations -fconcepts-diagnostics-depth=4 -ftemplate-depth=100860 -Ofast -flto -pipe -march=native -DDEBUG -DCUDA
OP            = -funsafe-math-optimizations -fconcepts-diagnostics-depth=4 -ftemplate-depth=100860 -Ofast -flto -pipe -march=skylake -DNDEBUG
OP            = -funsafe-math-optimizations -fconcepts-diagnostics-depth=4 -ftemplate-depth=100860 -Ofast -flto=auto  -funroll-all-loops -pipe -march=native -DNDEBUG

CXX           = g++
#CXX           = clang++
CXXFLAGS      = -std=c++20 -Wall -Wextra -fmax-errors=3 -ftemplate-backtrace-limit=0 -fdata-sections -ffunction-sections $(OP)

#LFLAGS        = $(OP) -L/opt/cuda/lib64 -pthread  -lcudart -lcublas -lstdc++fs ${LOP}
#LFLAGS        = $(OP) -L/opt/cuda/lib64 -pthread  -lcudart -lcublas -lstdc++fs -lc++abi ${LOP}
LFLAGS        = $(OP) -pthread  -lstdc++fs ${LOP}
LFLAGS        = $(OP) -L/opt/cuda/lib64 -pthread  -lstdc++fs -lcblas ${LOP}
LFLAGS        = $(OP) -L/opt/cuda/lib64 -pthread  -lcudart -lcublas -lstdc++fs ${LOP}
LFLAGS        = $(OP) -L/opt/cuda/lib64 -pthread  -lstdc++fs ${LOP}
LFLAGS        = $(OP) -pg -O0 -pthread  ${LOP}
LFLAGS        = $(OP) -L/opt/cuda/lib64 -pthread  -lcudart -lcublas -lstdc++fs ${LOP}

#CXX           = g++
#OP            = -O0  -pg -DDEBUG
#CXXFLAGS      = -std=c++2a -Wall -Wextra -fmax-errors=1 $(OP)
#LFLAGS        = $(OP) -pg -O0

LINK          = $(CXX) $(LFLAGS)

####### Output directory
OBJECTS_DIR   = ./obj
BIN_DIR       = ./bin
LIB_DIR       = .
LOG_DIR       = .

all: test

test: id tensor graph

tensor: test/tensor.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_tensor.o test/tensor.cc
	$(LINK) -o $(BIN_DIR)/test_tensor $(OBJECTS_DIR)/test_tensor.o $(LFLAGS)

tensor_glorot_uniform: test/tensor_glorot_uniform.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_tensor_glorot_uniform.o test/tensor_glorot_uniform.cc
	$(LINK) -o $(BIN_DIR)/test_tensor_glorot_uniform $(OBJECTS_DIR)/test_tensor_glorot_uniform.o $(LFLAGS)

id: test/id.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_id.o test/id.cc
	$(LINK) -o $(BIN_DIR)/test_id $(OBJECTS_DIR)/test_id.o $(LFLAGS)

range: test/range.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_range.o test/range.cc
	$(LINK) -o $(BIN_DIR)/test_range $(OBJECTS_DIR)/test_range.o $(LFLAGS)

graph: test/graph.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_graph.o test/graph.cc
	$(LINK) -o $(BIN_DIR)/test_graph $(OBJECTS_DIR)/test_graph.o $(LFLAGS)

optimize: test/optimize.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_optimize.o test/optimize.cc
	$(LINK) -o $(BIN_DIR)/test_optimize $(OBJECTS_DIR)/test_optimize.o $(LFLAGS)

xor: test/xor.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_xor.o test/xor.cc
	$(LINK) -o $(BIN_DIR)/test_xor $(OBJECTS_DIR)/test_xor.o $(LFLAGS)

sigmoid: test/sigmoid.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_sigmoid.o test/sigmoid.cc
	$(LINK) -o $(BIN_DIR)/test_sigmoid $(OBJECTS_DIR)/test_sigmoid.o $(LFLAGS)

softmax: test/softmax.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_softmax.o test/softmax.cc
	$(LINK) -o $(BIN_DIR)/test_softmax $(OBJECTS_DIR)/test_softmax.o $(LFLAGS)

tensor_add: test/tensor_add.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_tensor_add.o test/tensor_add.cc
	$(LINK) -o $(BIN_DIR)/test_tensor_add $(OBJECTS_DIR)/test_tensor_add.o $(LFLAGS)

buffered_allocator: test/buffered_allocator.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_buffered_allocator.o test/buffered_allocator.cc
	$(LINK) -o $(BIN_DIR)/test_buffered_allocator $(OBJECTS_DIR)/test_buffered_allocator.o $(LFLAGS)

tensor_io: test/tensor_io.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_tensor_io.o test/tensor_io.cc
	$(LINK) -o $(BIN_DIR)/test_tensor_io $(OBJECTS_DIR)/test_tensor_io.o $(LFLAGS)

mnist: test/mnist.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist.o test/mnist.cc
	$(LINK) -o $(BIN_DIR)/test_mnist $(OBJECTS_DIR)/test_mnist.o $(LFLAGS)

mnist_leaky_relu: test/mnist_leaky_relu.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_leaky_relu.o test/mnist_leaky_relu.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_leaky_relu $(OBJECTS_DIR)/test_mnist_leaky_relu.o $(LFLAGS)

img2col: test/img2col.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_img2col.o test/img2col.cc
	$(LINK) -o $(BIN_DIR)/test_img2col $(OBJECTS_DIR)/test_img2col.o $(LFLAGS)

transpose: test/transpose.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_transpose.o test/transpose.cc
	$(LINK) -o $(BIN_DIR)/test_transpose $(OBJECTS_DIR)/test_transpose.o $(LFLAGS)

conv2d: test/conv2d.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_conv2d.o test/conv2d.cc
	$(LINK) -o $(BIN_DIR)/test_conv2d $(OBJECTS_DIR)/test_conv2d.o $(LFLAGS)

reshape: test/reshape.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_reshape.o test/reshape.cc
	$(LINK) -o $(BIN_DIR)/test_reshape $(OBJECTS_DIR)/test_reshape.o $(LFLAGS)

flatten: test/flatten.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_flatten.o test/flatten.cc
	$(LINK) -o $(BIN_DIR)/test_flatten $(OBJECTS_DIR)/test_flatten.o $(LFLAGS)

overload: test/overload.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_overload.o test/overload.cc
	$(LINK) -o $(BIN_DIR)/test_overload $(OBJECTS_DIR)/test_overload.o $(LFLAGS)

drop_out: test/drop_out.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_drop_out.o test/drop_out.cc
	$(LINK) -o $(BIN_DIR)/test_drop_out $(OBJECTS_DIR)/test_drop_out.o $(LFLAGS)

max_pooling_2d: test/max_pooling_2d.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_max_pooling_2d.o test/max_pooling_2d.cc
	$(LINK) -o $(BIN_DIR)/test_max_pooling_2d $(OBJECTS_DIR)/test_max_pooling_2d.o $(LFLAGS)

mnist_conv2d: test/mnist_conv2d.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_conv2d.o test/mnist_conv2d.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_conv2d $(OBJECTS_DIR)/test_mnist_conv2d.o $(LFLAGS)

average_pooling_2d: test/average_pooling_2d.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_average_pooling_2d.o test/average_pooling_2d.cc
	$(LINK) -o $(BIN_DIR)/test_average_pooling_2d $(OBJECTS_DIR)/test_average_pooling_2d.o $(LFLAGS)

up_sampling_2d: test/up_sampling_2d.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_up_sampling_2d.o test/up_sampling_2d.cc
	$(LINK) -o $(BIN_DIR)/test_up_sampling_2d $(OBJECTS_DIR)/test_up_sampling_2d.o $(LFLAGS)

tensor_mm: test/tensor_mm.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_tensor_mm.o test/tensor_mm.cc
	$(LINK) -o $(BIN_DIR)/test_tensor_mm $(OBJECTS_DIR)/test_tensor_mm.o $(LFLAGS)

constant: test/constant.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_constant.o test/constant.cc
	$(LINK) -o $(BIN_DIR)/test_constant $(OBJECTS_DIR)/test_constant.o $(LFLAGS)

elementwise_multiply: test/elementwise_multiply.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_elementwise_multiply.o test/elementwise_multiply.cc
	$(LINK) -o $(BIN_DIR)/test_elementwise_multiply $(OBJECTS_DIR)/test_elementwise_multiply.o $(LFLAGS)

normalization: test/normalization.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_normalization.o test/normalization.cc
	$(LINK) -o $(BIN_DIR)/test_normalization $(OBJECTS_DIR)/test_normalization.o $(LFLAGS)

mnist_bn: test/mnist_bn.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_bn.o test/mnist_bn.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_bn $(OBJECTS_DIR)/test_mnist_bn.o $(LFLAGS)

cuda_memcpy: test/cuda_memcpy.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_cuda_memcpy.o test/cuda_memcpy.cc
	$(LINK) -o $(BIN_DIR)/test_cuda_memcpy $(OBJECTS_DIR)/test_cuda_memcpy.o $(LFLAGS)

timer: test/timer.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_timer.o test/timer.cc
	$(LINK) -o $(BIN_DIR)/test_timer $(OBJECTS_DIR)/test_timer.o $(LFLAGS)

imageio: test/imageio.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_imageio.o test/imageio.cc
	$(LINK) -o $(BIN_DIR)/test_imageio $(OBJECTS_DIR)/test_imageio.o $(LFLAGS)

keras_input: test/keras_input.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_keras_input.o test/keras_input.cc
	$(LINK) -o $(BIN_DIR)/test_keras_input $(OBJECTS_DIR)/test_keras_input.o $(LFLAGS)

float32: test/float32.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_float32.o test/float32.cc
	$(LINK) -o $(BIN_DIR)/test_float32 $(OBJECTS_DIR)/test_float32.o $(LFLAGS)

keras_initializer: test/keras_initializer.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_keras_initializer.o test/keras_initializer.cc
	$(LINK) -o $(BIN_DIR)/test_keras_initializer $(OBJECTS_DIR)/test_keras_initializer.o $(LFLAGS)

mnist_sgd: test/mnist_sgd.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_sgd.o test/mnist_sgd.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_sgd $(OBJECTS_DIR)/test_mnist_sgd.o $(LFLAGS)

mnist_adagrad: test/mnist_adagrad.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_adagrad.o test/mnist_adagrad.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_adagrad $(OBJECTS_DIR)/test_mnist_adagrad.o $(LFLAGS)

mnist_adam: test/mnist_adam.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_adam.o test/mnist_adam.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_adam $(OBJECTS_DIR)/test_mnist_adam.o $(LFLAGS)

mnist_rmsprop: test/mnist_rmsprop.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_rmsprop.o test/mnist_rmsprop.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_rmsprop $(OBJECTS_DIR)/test_mnist_rmsprop.o $(LFLAGS)

mnist_adadelta: test/mnist_adadelta.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_adadelta.o test/mnist_adadelta.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_adadelta $(OBJECTS_DIR)/test_mnist_adadelta.o $(LFLAGS)

glob: test/glob.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_glob.o test/glob.cc
	$(LINK) -o $(BIN_DIR)/test_glob $(OBJECTS_DIR)/test_glob.o $(LFLAGS)

keras_dense: test/keras_dense.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_keras_dense.o test/keras_dense.cc
	$(LINK) -o $(BIN_DIR)/test_keras_dense $(OBJECTS_DIR)/test_keras_dense.o $(LFLAGS)

keras_model: test/keras_model.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_keras_model.o test/keras_model.cc
	$(LINK) -o $(BIN_DIR)/test_keras_model $(OBJECTS_DIR)/test_keras_model.o $(LFLAGS)

reverse: test/reverse.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_reverse.o test/reverse.cc
	$(LINK) -o $(BIN_DIR)/test_reverse $(OBJECTS_DIR)/test_reverse.o $(LFLAGS)

tqdm: test/tqdm.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_tqdm.o test/tqdm.cc
	$(LINK) -o $(BIN_DIR)/test_tqdm $(OBJECTS_DIR)/test_tqdm.o $(LFLAGS)

mnist_keras: test/mnist_keras.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_keras.o test/mnist_keras.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_keras $(OBJECTS_DIR)/test_mnist_keras.o $(LFLAGS)

state: test/state.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_state.o test/state.cc
	$(LINK) -o $(BIN_DIR)/test_state $(OBJECTS_DIR)/test_state.o $(LFLAGS)

concatenate: test/concatenate.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_concatenate.o test/concatenate.cc
	$(LINK) -o $(BIN_DIR)/test_concatenate $(OBJECTS_DIR)/test_concatenate.o $(LFLAGS)

copy: test/copy.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_copy.o test/copy.cc
	$(LINK) -o $(BIN_DIR)/test_copy $(OBJECTS_DIR)/test_copy.o $(LFLAGS)

lstm_1: test/lstm_1.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_lstm_1.o test/lstm_1.cc
	$(LINK) -o $(BIN_DIR)/test_lstm_1 $(OBJECTS_DIR)/test_lstm_1.o $(LFLAGS)

vgg16: examples/vgg16/vgg16.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_vgg16.o examples/vgg16/vgg16.cc
	$(LINK) -o $(BIN_DIR)/test_vgg16 $(OBJECTS_DIR)/test_vgg16.o $(LFLAGS)

mnist_mini: test/mnist_mini.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_mini.o test/mnist_mini.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_mini $(OBJECTS_DIR)/test_mnist_mini.o $(LFLAGS)

mnist_conv2d_mini: test/mnist_conv2d_mini.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_conv2d_mini.o test/mnist_conv2d_mini.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_conv2d_mini $(OBJECTS_DIR)/test_mnist_conv2d_mini.o $(LFLAGS)

mnist_conv2d_mini_negative: test/mnist_conv2d_mini_negative.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_conv2d_mini_negative.o test/mnist_conv2d_mini_negative.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_conv2d_mini_negative $(OBJECTS_DIR)/test_mnist_conv2d_mini_negative.o $(LFLAGS)

mnist_bn_mini: test/mnist_bn_mini.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_bn_mini.o test/mnist_bn_mini.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_bn_mini $(OBJECTS_DIR)/test_mnist_bn_mini.o $(LFLAGS)

session: test/session.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_session.o test/session.cc
	$(LINK) -o $(BIN_DIR)/test_session $(OBJECTS_DIR)/test_session.o $(LFLAGS)

layer_plus: test/layer_plus.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_plus.o test/layer_plus.cc
	$(LINK) -o $(BIN_DIR)/test_layer_plus $(OBJECTS_DIR)/test_layer_plus.o $(LFLAGS)

layer_multiply: test/layer_multiply.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_multiply.o test/layer_multiply.cc
	$(LINK) -o $(BIN_DIR)/test_layer_multiply $(OBJECTS_DIR)/test_layer_multiply.o $(LFLAGS)

layer_log: test/layer_log.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_log.o test/layer_log.cc
	$(LINK) -o $(BIN_DIR)/test_layer_log $(OBJECTS_DIR)/test_layer_log.o $(LFLAGS)

layer_softmax: test/layer_softmax.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_softmax.o test/layer_softmax.cc
	$(LINK) -o $(BIN_DIR)/test_layer_softmax $(OBJECTS_DIR)/test_layer_softmax.o $(LFLAGS)

layer_mae: test/layer_mae.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_mae.o test/layer_mae.cc
	$(LINK) -o $(BIN_DIR)/test_layer_mae $(OBJECTS_DIR)/test_layer_mae.o $(LFLAGS)

conv2d_se: test/conv2d_se.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_conv2d_se.o test/conv2d_se.cc
	$(LINK) -o $(BIN_DIR)/test_conv2d_se $(OBJECTS_DIR)/test_conv2d_se.o $(LFLAGS)

maximum: test/maximum.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_maximum.o test/maximum.cc
	$(LINK) -o $(BIN_DIR)/test_maximum $(OBJECTS_DIR)/test_maximum.o $(LFLAGS)

value: test/value.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_value.o test/value.cc
	$(LINK) -o $(BIN_DIR)/test_value $(OBJECTS_DIR)/test_value.o $(LFLAGS)

hinge_loss: test/hinge_loss.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_hinge_loss.o test/hinge_loss.cc
	$(LINK) -o $(BIN_DIR)/test_hinge_loss $(OBJECTS_DIR)/test_hinge_loss.o $(LFLAGS)

mnist_restore: test/mnist_restore.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_restore.o test/mnist_restore.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_restore $(OBJECTS_DIR)/test_mnist_restore.o $(LFLAGS)

mnist_model: test/mnist_model.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_model.o test/mnist_model.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_model $(OBJECTS_DIR)/test_mnist_model.o $(LFLAGS)

layer_random_normal_like: test/layer_random_normal_like.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_random_normal_like.o test/layer_random_normal_like.cc
	$(LINK) -o $(BIN_DIR)/test_layer_random_normal_like $(OBJECTS_DIR)/test_layer_random_normal_like.o $(LFLAGS)

mnist_vae: test/mnist_vae.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_vae.o test/mnist_vae.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_vae $(OBJECTS_DIR)/test_mnist_vae.o $(LFLAGS)

layer_exp: test/layer_exp.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_exp.o test/layer_exp.cc
	$(LINK) -o $(BIN_DIR)/test_layer_exp $(OBJECTS_DIR)/test_layer_exp.o $(LFLAGS)

model: test/model.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_model.o test/model.cc
	$(LINK) -o $(BIN_DIR)/test_model $(OBJECTS_DIR)/test_model.o $(LFLAGS)

mnist_autoencoder: test/mnist_autoencoder.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_autoencoder.o test/mnist_autoencoder.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_autoencoder $(OBJECTS_DIR)/test_mnist_autoencoder.o $(LFLAGS)

mnist_compiled_model: test/mnist_compiled_model.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_compiled_model.o test/mnist_compiled_model.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_compiled_model $(OBJECTS_DIR)/test_mnist_compiled_model.o $(LFLAGS)

mnist_model_fit: test/mnist_model_fit.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_model_fit.o test/mnist_model_fit.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_model_fit $(OBJECTS_DIR)/test_mnist_model_fit.o $(LFLAGS)

mnist_dataset: test/mnist_dataset.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_dataset.o test/mnist_dataset.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_dataset $(OBJECTS_DIR)/test_mnist_dataset.o $(LFLAGS)

mnist_minimal: test/mnist_minimal.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_minimal.o test/mnist_minimal.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_minimal $(OBJECTS_DIR)/test_mnist_minimal.o $(LFLAGS)

mnist_conv2d_minimal: test/mnist_conv2d_minimal.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_conv2d_minimal.o test/mnist_conv2d_minimal.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_conv2d_minimal $(OBJECTS_DIR)/test_mnist_conv2d_minimal.o $(LFLAGS)

context: test/context.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_context.o test/context.cc
	$(LINK) -o $(BIN_DIR)/test_context $(OBJECTS_DIR)/test_context.o $(LFLAGS)

model_trainable: test/model_trainable.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_model_trainable.o test/model_trainable.cc
	$(LINK) -o $(BIN_DIR)/test_model_trainable $(OBJECTS_DIR)/test_model_trainable.o $(LFLAGS)

#dcgan: examples/dcgan.cc
#	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/example_dcgan.o examples/dcgan.cc
#	$(LINK) -o $(BIN_DIR)/example_dcgan $(OBJECTS_DIR)/example_dcgan.o $(LFLAGS)

fashion_mnist_dataset: test/fashion_mnist_dataset.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_fashion_mnist_dataset.o test/fashion_mnist_dataset.cc
	$(LINK) -o $(BIN_DIR)/test_fashion_mnist_dataset $(OBJECTS_DIR)/test_fashion_mnist_dataset.o $(LFLAGS)

equal: test/equal.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_equal.o test/equal.cc
	$(LINK) -o $(BIN_DIR)/test_equal $(OBJECTS_DIR)/test_equal.o $(LFLAGS)

sign: test/sign.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_sign.o test/sign.cc
	$(LINK) -o $(BIN_DIR)/test_sign $(OBJECTS_DIR)/test_sign.o $(LFLAGS)

mnist_ln_mini: test/mnist_ln_mini.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_ln_mini.o test/mnist_ln_mini.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_ln_mini $(OBJECTS_DIR)/test_mnist_ln_mini.o $(LFLAGS)

dcgan: test/dcgan.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_dcgan.o test/dcgan.cc
	$(LINK) -o $(BIN_DIR)/test_dcgan $(OBJECTS_DIR)/test_dcgan.o $(LFLAGS)

computation_graph_dump: test/computation_graph_dump.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_computation_graph_dump.o test/computation_graph_dump.cc
	$(LINK) -o $(BIN_DIR)/test_computation_graph_dump $(OBJECTS_DIR)/test_computation_graph_dump.o $(LFLAGS)

layer_selu: test/layer_selu.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_selu.o test/layer_selu.cc
	$(LINK) -o $(BIN_DIR)/test_layer_selu $(OBJECTS_DIR)/test_layer_selu.o $(LFLAGS)

layer_zeropadding2d: test/layer_zeropadding2d.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_zeropadding2d.o test/layer_zeropadding2d.cc
	$(LINK) -o $(BIN_DIR)/test_layer_zeropadding2d $(OBJECTS_DIR)/test_layer_zeropadding2d.o $(LFLAGS)

mnist_regularized: test/mnist_regularized.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_regularized.o test/mnist_regularized.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_regularized $(OBJECTS_DIR)/test_mnist_regularized.o $(LFLAGS)

unet: examples/unet/unet.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_unet.o examples/unet/unet.cc
	$(LINK) -o $(BIN_DIR)/test_unet $(OBJECTS_DIR)/test_unet.o $(LFLAGS)

mnist_conv2d_visualization: test/mnist_conv2d_visualization.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_conv2d_visualization.o test/mnist_conv2d_visualization.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_conv2d_visualization $(OBJECTS_DIR)/test_mnist_conv2d_visualization.o $(LFLAGS)

mnist_tsne: test/mnist_tsne.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_tsne.o test/mnist_tsne.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_tsne $(OBJECTS_DIR)/test_mnist_tsne.o $(LFLAGS)

layer_repeat: test/layer_repeat.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_repeat.o test/layer_repeat.cc
	$(LINK) -o $(BIN_DIR)/test_layer_repeat $(OBJECTS_DIR)/test_layer_repeat.o $(LFLAGS)

layer_reduce_min: test/layer_reduce_min.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_reduce_min.o test/layer_reduce_min.cc
	$(LINK) -o $(BIN_DIR)/test_layer_reduce_min $(OBJECTS_DIR)/test_layer_reduce_min.o $(LFLAGS)

layer_reduce_max: test/layer_reduce_max.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_reduce_max.o test/layer_reduce_max.cc
	$(LINK) -o $(BIN_DIR)/test_layer_reduce_max $(OBJECTS_DIR)/test_layer_reduce_max.o $(LFLAGS)

layer_reduce_sum: test/layer_reduce_sum.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_reduce_sum.o test/layer_reduce_sum.cc
	$(LINK) -o $(BIN_DIR)/test_layer_reduce_sum $(OBJECTS_DIR)/test_layer_reduce_sum.o $(LFLAGS)

layer_hypot: test/layer_hypot.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_hypot.o test/layer_hypot.cc
	$(LINK) -o $(BIN_DIR)/test_layer_hypot $(OBJECTS_DIR)/test_layer_hypot.o $(LFLAGS)

complex_abs: test/complex_abs.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_complex_abs.o test/complex_abs.cc
	$(LINK) -o $(BIN_DIR)/test_complex_abs $(OBJECTS_DIR)/test_complex_abs.o $(LFLAGS)

complex_multiplication: test/complex_multiplication.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_complex_multiplication.o test/complex_multiplication.cc
	$(LINK) -o $(BIN_DIR)/test_complex_multiplication $(OBJECTS_DIR)/test_complex_multiplication.o $(LFLAGS)

layer_atan2: test/layer_atan2.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_atan2.o test/layer_atan2.cc
	$(LINK) -o $(BIN_DIR)/test_layer_atan2 $(OBJECTS_DIR)/test_layer_atan2.o $(LFLAGS)

complex_arg: test/complex_arg.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_complex_arg.o test/complex_arg.cc
	$(LINK) -o $(BIN_DIR)/test_complex_arg $(OBJECTS_DIR)/test_complex_arg.o $(LFLAGS)

layer_cos: test/layer_cos.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_cos.o test/layer_cos.cc
	$(LINK) -o $(BIN_DIR)/test_layer_cos $(OBJECTS_DIR)/test_layer_cos.o $(LFLAGS)

layer_tanh: test/layer_tanh.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_tanh.o test/layer_tanh.cc
	$(LINK) -o $(BIN_DIR)/test_layer_tanh $(OBJECTS_DIR)/test_layer_tanh.o $(LFLAGS)

layer_relu6: test/layer_relu6.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_relu6.o test/layer_relu6.cc
	$(LINK) -o $(BIN_DIR)/test_layer_relu6 $(OBJECTS_DIR)/test_layer_relu6.o $(LFLAGS)

layer_assign: test/layer_assign.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_assign.o test/layer_assign.cc
	$(LINK) -o $(BIN_DIR)/test_layer_assign $(OBJECTS_DIR)/test_layer_assign.o $(LFLAGS)

layer_erf: test/layer_erf.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_erf.o test/layer_erf.cc
	$(LINK) -o $(BIN_DIR)/test_layer_erf $(OBJECTS_DIR)/test_layer_erf.o $(LFLAGS)

layer_gaussian: test/layer_gaussian.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_gaussian.o test/layer_gaussian.cc
	$(LINK) -o $(BIN_DIR)/test_layer_gaussian $(OBJECTS_DIR)/test_layer_gaussian.o $(LFLAGS)

layer_cropping2d: test/layer_cropping2d.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_cropping2d.o test/layer_cropping2d.cc
	$(LINK) -o $(BIN_DIR)/test_layer_cropping2d $(OBJECTS_DIR)/test_layer_cropping2d.o $(LFLAGS)

layer_sliding2d: test/layer_sliding2d.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_sliding2d.o test/layer_sliding2d.cc
	$(LINK) -o $(BIN_DIR)/test_layer_sliding2d $(OBJECTS_DIR)/test_layer_sliding2d.o $(LFLAGS)

mnist_save_restore: test/mnist_save_restore.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_save_restore.o test/mnist_save_restore.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_save_restore $(OBJECTS_DIR)/test_mnist_save_restore.o $(LFLAGS)

mnist_duplicated: test/mnist_duplicated.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_duplicated.o test/mnist_duplicated.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_duplicated $(OBJECTS_DIR)/test_mnist_duplicated.o $(LFLAGS)

mnist_conv2d_sliding: test/mnist_conv2d_sliding.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_conv2d_sliding.o test/mnist_conv2d_sliding.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_conv2d_sliding $(OBJECTS_DIR)/test_mnist_conv2d_sliding.o $(LFLAGS)

mnist_ls: test/mnist_ls.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_ls.o test/mnist_ls.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_ls $(OBJECTS_DIR)/test_mnist_ls.o $(LFLAGS)

for_each: test/for_each.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_for_each.o test/for_each.cc
	$(LINK) -o $(BIN_DIR)/test_for_each $(OBJECTS_DIR)/test_for_each.o $(LFLAGS)

xtensor: test/xtensor.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_xtensor.o test/xtensor.cc
	$(LINK) -o $(BIN_DIR)/test_xtensor $(OBJECTS_DIR)/test_xtensor.o $(LFLAGS)

mnist_debug: test/mnist_debug.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_mnist_debug.o test/mnist_debug.cc
	$(LINK) -o $(BIN_DIR)/test_mnist_debug $(OBJECTS_DIR)/test_mnist_debug.o $(LFLAGS)

layer_sum: test/layer_sum.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_sum.o test/layer_sum.cc
	$(LINK) -o $(BIN_DIR)/test_layer_sum $(OBJECTS_DIR)/test_layer_sum.o $(LFLAGS)
field: test/field.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_field.o test/field.cc
	$(LINK) -o $(BIN_DIR)/test_field $(OBJECTS_DIR)/test_field.o $(LFLAGS)

keras_layer: test/keras_layer.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_keras_layer.o test/keras_layer.cc
	$(LINK) -o $(BIN_DIR)/test_keras_layer $(OBJECTS_DIR)/test_keras_layer.o $(LFLAGS)

tensor_poisson: test/tensor_poisson.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_tensor_poisson.o test/tensor_poisson.cc
	$(LINK) -o $(BIN_DIR)/test_tensor_poisson $(OBJECTS_DIR)/test_tensor_poisson.o $(LFLAGS)

denoise_poisson: test/denoise_poisson.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_denoise_poisson.o test/denoise_poisson.cc
	$(LINK) -o $(BIN_DIR)/test_denoise_poisson $(OBJECTS_DIR)/test_denoise_poisson.o $(LFLAGS)

layer_divide: test/layer_divide.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_divide.o test/layer_divide.cc
	$(LINK) -o $(BIN_DIR)/test_layer_divide $(OBJECTS_DIR)/test_layer_divide.o $(LFLAGS)

layer_expand_dims: test/layer_expand_dims.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_expand_dims.o test/layer_expand_dims.cc
	$(LINK) -o $(BIN_DIR)/test_layer_expand_dims $(OBJECTS_DIR)/test_layer_expand_dims.o $(LFLAGS)

layer_argmax: test/layer_argmax.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_argmax.o test/layer_argmax.cc
	$(LINK) -o $(BIN_DIR)/test_layer_argmax $(OBJECTS_DIR)/test_layer_argmax.o $(LFLAGS)

layer_argmin: test/layer_argmin.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_argmin.o test/layer_argmin.cc
	$(LINK) -o $(BIN_DIR)/test_layer_argmin $(OBJECTS_DIR)/test_layer_argmin.o $(LFLAGS)

yolov3: examples/yolov3/yolov3.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_yolov3.o examples/yolov3/yolov3.cc
	$(LINK) -o $(BIN_DIR)/test_yolov3 $(OBJECTS_DIR)/test_yolov3.o $(LFLAGS)

gemm_n2: test/gemm_n2.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_gemm_n2.o test/gemm_n2.cc
	$(LINK) -o $(BIN_DIR)/test_gemm_n2 $(OBJECTS_DIR)/test_gemm_n2.o $(LFLAGS)

layer_pow: test/layer_pow.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_layer_pow.o test/layer_pow.cc
	$(LINK) -o $(BIN_DIR)/test_layer_pow $(OBJECTS_DIR)/test_layer_pow.o $(LFLAGS)


.PHONY: clean clean_obj clean_bin clean_misc
clean: clean_obj clean_bin clean_misc
clean_obj:
	-rm $(OBJECTS_DIR)/*.o
clean_bin:
	-rm $(BIN_DIR)/*
clean_misc:
	-rm ./*.png
	-rm ./*.dot
	-rm ./*.session

