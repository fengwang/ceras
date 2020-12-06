CXX           = clang++
OP            = -funsafe-math-optimizations  -Ofast -flto -pipe -march=native -DNDEBUG
CXXFLAGS      = -std=c++2a -Wall -Wextra -ferror-limit=3 -ftemplate-backtrace-limit=0 $(OP)
LFLAGS        = $(OP) -pthread
LINK          = $(CXX)

####### Output directory
OBJECTS_DIR   = ./obj
BIN_DIR       = ./bin
LIB_DIR       = .
LOG_DIR       = .

all: mnist

mnist: ./mnist.cc
	$(CXX) -c $(CXXFLAGS) -o $(OBJECTS_DIR)/example_mnist.o ./mnist.cc
	$(LINK) -o $(BIN_DIR)/example_mnist $(OBJECTS_DIR)/example_mnist.o $(LFLAGS)

.PHONY: clean clean_obj clean_bin
clean: clean_obj clean_bin
clean_obj:
	rm $(OBJECTS_DIR)/*.o
clean_bin:
	rm $(BIN_DIR)/*

