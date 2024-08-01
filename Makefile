CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -Wextra -O3
NVCCFLAGS = -std=c++14 -O3
INCLUDES = -Iinclude

.PHONY: all clean test

all: build/pdto_cpp

build/pdto_cpp: src/main.cpp src/decision_tree.cpp src/parallel_cpu.cpp src/parallel_gpu.cu
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@

test: build/test_decision_tree build/test_parallel_cpu build/test_parallel_gpu
	./build/test_decision_tree
	./build/test_parallel_cpu
	./build/test_parallel_gpu

build/test_decision_tree: tests/test_decision_tree.cpp src/decision_tree.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

build/test_parallel_cpu: tests/test_parallel_cpu.cpp src/parallel_cpu.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

build/test_parallel_gpu: tests/test_parallel_gpu.cu src/parallel_gpu.cu
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@

clean:
	rm -rf build
	find . -name "*.o" -type f -delete
	find . -name "*.pyc" -type f -delete