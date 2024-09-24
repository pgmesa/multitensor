# MultiTensor

**MultiTensor** is a project that implements multidimensional tensor operations in C and C++, with Python bindings. It offers a custom implementation of N-dimensional strided tensors and various tensor operations, similar to those in libraries like NumPy and PyTorch, but built from scratch.

The project explores the technical aspects of developing high-performance numerical computing libraries and the methods for exposing low-level implementations to high-level languages like Python. It serves as a guide for understanding the inner workings of low-level tensor operations, memory management, and the process of creating language interfaces between C/C++ and Python.

## Project Overview

- Custom implementation of N-dimensional strided tensors in C and C++
- Support for multiple data types (uint8, int32, float32)
- Tensor operations (e.g., element-wise operations, N-dimensional matrix multiplication, indexing)
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
   git clone https://github.com/yourusername/multitensor.git
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

## Future Developments

- Implement more tensor operations (convolutions, broadcasting, slicing)
- Add support for GPU acceleration
- Expand test suite and benchmarking tools
- Extend data type support to include more types (e.g., float16, complex)