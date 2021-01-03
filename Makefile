#CXX           = clang++
#OP            = -funsafe-math-optimizations  -Ofast -flto -pipe -march=native -DDEBUG
#CXXFLAGS      = -std=c++2a -Wall -Wextra -ferror-limit=1 -ftemplate-backtrace-limit=0 $(OP)
#LFLAGS        = $(OP) -pthread

CXX           = g++
OP            = -O0  -pg -DDEBUG
CXXFLAGS      = -std=c++2a -Wall -Wextra $(OP)
LFLAGS        = $(OP) -pg -O0

LINK          = $(CXX)

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

xor_se: test/xor_se.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/test_xor_se.o test/xor_se.cc
	$(LINK) -o $(BIN_DIR)/test_xor_se $(OBJECTS_DIR)/test_xor_se.o $(LFLAGS)


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


.PHONY: clean clean_obj clean_bin
clean: clean_obj clean_bin
clean_obj:
	rm $(OBJECTS_DIR)/*.o
clean_bin:
	rm $(BIN_DIR)/*

