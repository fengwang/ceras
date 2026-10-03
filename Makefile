# Maintained tests use the explicit CMake manifest. Historical demos are not CI gates.
CXX ?= c++
DEBUG ?= 1
CUDA ?= 0
CBLAS ?= 0
BUILD_DIR ?= build/make
BUILD_TYPE := $(if $(filter 1,$(DEBUG)),Debug,Release)
.PHONY: all configure build test ci clean gemm image mnist benchmarks
all: test
configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_CXX_COMPILER=$(CXX) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCERAS_ENABLE_CUDA=$(if $(filter 1,$(CUDA)),ON,OFF) -DCERAS_ENABLE_CBLAS=$(if $(filter 1,$(CBLAS)),ON,OFF) $(CMAKE_ARGS)
build: configure
	cmake --build $(BUILD_DIR) --parallel 2
test ci: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure --no-tests=error
gemm image: build
	ctest --test-dir $(BUILD_DIR) -R '^$@$$' --output-on-failure --no-tests=error
mnist: CMAKE_ARGS += -DCERAS_MNIST_DIR=$(CURDIR)/dataset/mnist
mnist: build
	ctest --test-dir $(BUILD_DIR) -R '^mnist_integration$$' --output-on-failure --no-tests=error
benchmarks: configure
	cmake --build $(BUILD_DIR) --target benchmarks --parallel 2
clean:
	cmake --build $(BUILD_DIR) --target clean
