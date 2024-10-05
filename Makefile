
c_compiler=gcc
cpp_compiler=g++
src_dir=./C++/cpptensor

all: tensor_c tensor_cpp

tensor_c:
	$(c_compiler) ./C/tensor.c -o tensor_c

tensor_cpp:
	$(cpp_compiler) ./C++/main.cpp $(src_dir)/tensor.cpp $(src_dir)/utils.cpp $(src_dir)/dtype.cpp -o tensor_cpp

clean:
	del tensor_c*, tensor_cpp*