
c_compiler=gcc
cpp_compiler=g++

all: tensor_c tensor_cpp

tensor_c: ./C/tensor.c
	$(c_compiler) ./C/tensor.c -o tensor_c

tensor_cpp: ./C++/tensor.cpp
	$(cpp_compiler) ./C++/tensor.cpp -o tensor_cpp

clean:
	del tensor_c*, tensor_cpp*