# MultiTensor

**MultiTensor** is a project that implements multidimensional tensor operations in C and C++, with Python bindings. It provides a custom implementation of N-dimensional strided tensors and various tensor operations similar to those in libraries like NumPy and PyTorch. However, unlike production-ready libraries, **MultiTensor** is built from scratch to offer a deeper understanding of how low-level tensor operations can be implemented and exposed to high-level languages like Python.

## Purpose and Limitations

This project is **not designed to be a high-performance library** optimized with advanced techniques such as SIMD instructions, multi-threading, or leveraging optimized low-level libraries like OpenBLAS or MKL. While such optimizations are crucial for performance in production-level numerical computing libraries, **MultiTensor** is focused on exploring the fundamental aspects of building a tensor library from the ground up.

The primary goal of **MultiTensor** is to serve as a learning tool, offering insights into the process of implementing core tensor functionalities, including memory management, N-dimensional strided tensors, and basic element-wise operations. It also demonstrates how to expose these low-level implementations to Python through custom bindings, making it a valuable reference for anyone interested in the inner workings of tensor computation libraries. This project emphasizes functionality and correctness over performance, making it ideal for educational purposes, rather than high-performance computing.

## Project Overview

- Custom implementation of N-dimensional strided tensors in C and C++
- Support for multiple data types (uint8, int32, float32)
- Tensor operations (e.g., element-wise operations, N-dimensional matrix multiplication, broadcasting)
- Python bindings using Cython (for C) and pybind11 (for C++)
- Performance comparisons with NumPy NDArrays

## Requirements

- Python >= 3.9
- C compiler (gcc recommended)
- C++ compiler (g++ recommended)
- CMake

## Project Structure

1. Clone the repository:
   ```
   git clone https://github.com/pgmesa/multitensor.git
   cd multitensor
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Implementations

### C and C++ Native Implementations

The core tensor operations are implemented in both C and C++ for comparison:

To compile both implementations:
```
make
```

For C implementation only:
```
make tensor_c
```

For C++ implementation only:
```
make tensor_cpp
```

### Python Bindings

#### C Library (using Cython)

1. Navigate to the `C` directory:
   ```
   cd C
   ```

2. Compile the Cython wrapper:
   ```
   make build
   ```
   or
   ```
   python ./setup.py build_ext --inplace
   ```

3. Run the Python tests:
   ```
   make run
   ```
   or
   ```
   python test.py
   ```

#### C++ Library (using pybind11)

1. Navigate to the `C++` directory:
   ```
   cd C++
   ```

2. Build pybind11 bindings (Windows version):
   ```
   make build
   ```
   This command generates Visual Studio solution files for the pybind11 project using CMake, builds the project with MSBuild, and then copies the resulting .pyd file from the build directory to the current directory for use in Python.

3. Run the Python tests:
   ```
   make run
   ```
   or
   ```
   python test.py
   ```