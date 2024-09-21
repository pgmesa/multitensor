
import time
import numpy as np
from cytensor import CyTensor, UINT8, INT32, FLOAT32

# NOTE: 
# With integer types, the C matmul implementation and NumPy matmul perform 
# very similarly, but when using floating-point types, NumPy far outperforms 
# the current implementation for large matrices.
DATATYPE = INT32

# Basic example to demonstrate printing and conversion to numpy
def basic_example():
    shape1 = [2, 5, 8] 
    shape2 = [2, 8, 5] 

    t1 = CyTensor.ones(shape1, DATATYPE)
    t1_0 = t1[0] * 2
    t1_0 += 1.5
    t2 = CyTensor.fill(shape2, DATATYPE, value=4)  

    t3 = t1 @ t2

    # Print Cython tensor and convert it to NumPy
    print("Cython matrix multiplication result:")
    t3.print_tensor()

    np_array: np.ndarray = t3.to_numpy()
    print("Cython result as NumPy array:")
    print(np_array)
    print("Shape:", np_array.shape, "| Dtype:", np_array.dtype, "| Strides:", np_array.strides)


# Performance comparison for small matrices over multiple runs
def compare_small_matrices(num_runs=10000):
    shape1 = [1, 2, 5, 8]
    shape2 = [1, 2, 8, 5]

    # Prepare CyTensor objects
    t1 = CyTensor.fill(shape1, DATATYPE, value=2)
    t2 = CyTensor.fill(shape2, DATATYPE, value=4)

    # Perform multiple runs of Cython matrix multiplication
    start_time = time.time()
    for _ in range(num_runs):
        t3 = t1 @ t2
    end_time = time.time()
    cython_time = end_time - start_time

    # Perform multiple runs of NumPy matrix multiplication
    np_t1 = np.full(shape1, 2, dtype=_get_numpy_dtype(DATATYPE))
    np_t2 = np.full(shape2, 4, dtype=_get_numpy_dtype(DATATYPE))
    
    start_time = time.time()
    for _ in range(num_runs):
        np_result = np.matmul(np_t1, np_t2)
    end_time = time.time()
    numpy_time = end_time - start_time

    print(f"\nResults for {num_runs} small matrix multiplications:")
    print(f"Cython time: {cython_time:.6f} seconds")
    print(f"NumPy time: {numpy_time:.6f} seconds")
    print(f"Cython is {'faster' if cython_time < numpy_time else 'slower'} than NumPy by a factor of {numpy_time / cython_time:.2f}")


# Performance comparison for larger matrices with fewer runs
def compare_large_matrices(num_runs=10):
    shape1 = [10, 20, 50, 80]
    shape2 = [10, 20, 80, 50]

    # Prepare CyTensor objects
    t1 = CyTensor.fill(shape1, DATATYPE, value=2)
    t2 = CyTensor.fill(shape2, DATATYPE, value=4)

    # Perform fewer runs of Cython matrix multiplication
    start_time = time.time()
    for _ in range(num_runs):
        t3 = CyTensor.matmul(t1, t2)
    end_time = time.time()
    cython_time = end_time - start_time

    # Perform fewer runs of NumPy matrix multiplication
    np_t1 = np.full(shape1, 2, dtype=_get_numpy_dtype(DATATYPE))
    np_t2 = np.full(shape2, 4, dtype=_get_numpy_dtype(DATATYPE))

    start_time = time.time()
    for _ in range(num_runs):
        np_result = np.matmul(np_t1, np_t2)
    end_time = time.time()
    numpy_time = end_time - start_time

    print(f"\nResults for {num_runs} large matrix multiplications:")
    print(f"Cython time: {cython_time:.6f} seconds")
    print(f"NumPy time: {numpy_time:.6f} seconds")
    print(f"Cython is {'faster' if cython_time < numpy_time else 'slower'} than NumPy by a factor of {numpy_time / cython_time:.2f}")


# Helper function to map CyTensor data types to NumPy data types
def _get_numpy_dtype(dtype):
    if dtype == UINT8:
        return np.uint8
    elif dtype == INT32:
        return np.int32
    elif dtype == FLOAT32:
        return np.float32
    else:
        raise ValueError("Unsupported data type")


basic_example()
compare_small_matrices(num_runs=10000)
compare_large_matrices(num_runs=10)
