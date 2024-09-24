
import time
import numpy as np
from cpptensor import Tensor, DataType

# NOTE: 
# With integer types, the C++ matmul implementation and NumPy matmul perform 
# very similarly, but when using floating-point types, NumPy far outperforms 
# the current implementation for large matrices.
DATATYPE = DataType.INT32


def basic_example():
    shape1 = [2, 5, 8] 
    shape2 = [2, 8, 5] 

    t1 = Tensor.ones(shape1, DATATYPE)
    t1_0 = t1[0]
    t1_0 *= 2
    t1_0 += 1.5
    t2 = Tensor.fill(shape2, DATATYPE, 4.0)

    t3 = t1 @ t2

    print("CppTensor matrix multiplication result:")
    print(t3)


# Performance comparison for small matrices over multiple runs
def compare_small_matrices(num_runs=10000):
    shape1 = [1, 2, 5, 8]
    shape2 = [1, 2, 8, 5]

    t1 = Tensor.fill(shape1, DATATYPE, 2.0)
    t2 = Tensor.fill(shape2, DATATYPE, 4.0)

    start_time = time.time()
    for _ in range(num_runs):
        t3 = t1 @ t2
    end_time = time.time()
    cpp_time = end_time - start_time

    np_t1 = np.full(shape1, 2, dtype=_get_numpy_dtype(DATATYPE))
    np_t2 = np.full(shape2, 4, dtype=_get_numpy_dtype(DATATYPE))
    
    start_time = time.time()
    for _ in range(num_runs):
        np_result = np.matmul(np_t1, np_t2)
    end_time = time.time()
    numpy_time = end_time - start_time

    print(f"\nResults for {num_runs} small matrix multiplications:")
    print(f"CppTensor time: {cpp_time:.6f} seconds")
    print(f"NumPy time: {numpy_time:.6f} seconds")
    print(f"CppTensor is {'faster' if cpp_time < numpy_time else 'slower'} than NumPy by a factor of {numpy_time / cpp_time:.2f}")


# Performance comparison for larger matrices with fewer runs
def compare_large_matrices(num_runs=10):
    shape1 = [10, 20, 50, 80]
    shape2 = [10, 20, 80, 50]

    t1 = Tensor.fill(shape1, DATATYPE, 2.0)
    t2 = Tensor.fill(shape2, DATATYPE, 4.0)

    start_time = time.time()
    for _ in range(num_runs):
        t3 = Tensor.matmul(t1, t2)
    end_time = time.time()
    cpp_time = end_time - start_time

    np_t1 = np.full(shape1, 2, dtype=_get_numpy_dtype(DATATYPE))
    np_t2 = np.full(shape2, 4, dtype=_get_numpy_dtype(DATATYPE))

    start_time = time.time()
    for _ in range(num_runs):
        np_result = np.matmul(np_t1, np_t2)
    end_time = time.time()
    numpy_time = end_time - start_time

    print(f"\nResults for {num_runs} large matrix multiplications:")
    print(f"CppTensor time: {cpp_time:.6f} seconds")
    print(f"NumPy time: {numpy_time:.6f} seconds")
    print(f"CppTensor is {'faster' if cpp_time < numpy_time else 'slower'} than NumPy by a factor of {numpy_time / cpp_time:.2f}")


# Helper function to map CppTensor data types to NumPy data types
def _get_numpy_dtype(dtype):
    if dtype == DataType.UINT8:
        return np.uint8
    elif dtype == DataType.INT32:
        return np.int32
    elif dtype == DataType.FLOAT32:
        return np.float32
    else:
        raise ValueError("Unsupported data type")


basic_example()
compare_small_matrices(num_runs=10000)
compare_large_matrices(num_runs=10)
