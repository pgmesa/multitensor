# cython: language_level=3
from typing import Union
import numpy as np
# Import necessary Cython and C functions
from libc.stdlib cimport malloc
from libc.stdint cimport uint8_t, int32_t


UINT8 = 0
INT32 = 1
FLOAT32 = 2

cdef extern from "tensor.h":
    # Declare the C functions to use in Python
    ctypedef uint8_t uint8
    ctypedef int32_t int32
    ctypedef float float32

    ctypedef enum DataType:
        UINT8_t
        INT32_t
        FLOAT32_t

    ctypedef struct Shape:
        int32 *values
        int32 ndim

    ctypedef struct Tensor:
        void *data
        size_t numel
        Shape shape
        int32 *strides
        int ndim
        DataType dtype
        int is_view

    Tensor fill(Shape shape, DataType dtype, double value)
    Tensor empty(Shape shape, DataType dtype)
    Tensor zeros(Shape shape, DataType dtype)
    Tensor ones(Shape shape, DataType dtype)
    Tensor print_tensor(const Tensor *t1)
    void free_tensor(Tensor *tensor)

    Tensor add(Tensor *t1, Tensor *t2)
    void add_value(Tensor *t1, double value)
    Tensor mul(Tensor *t1, Tensor *t2)
    void mul_value(Tensor *t1, double value)
    Tensor matmul(Tensor *t1, Tensor *t2)
    Tensor tensor_index(Tensor *tensor, int idx)


# Wrapping C types for use in Python
cdef class CyTensor:
    cdef Tensor ctensor

    def __init__(self) -> None:
        pass

    def __cinit__(self):
        pass

    @staticmethod
    cdef Shape __build_shape(list shape):        
        cdef int32 ndim = <int32> len(shape)
        cdef int32 dtype_size = <int32> sizeof(int32)
        cdef int32* cshape = <int32*> malloc(ndim * dtype_size)
        for i in range(ndim):
            cshape[i] = shape[i]

        cdef Shape c_shape
        c_shape.values = cshape
        c_shape.ndim = ndim
        
        return c_shape

    @staticmethod
    def fill(shape:list, dtype:int, value:float) -> CyTensor:
        cdef Shape cshape = CyTensor.__build_shape(shape)
        cdef CyTensor tensor = CyTensor()
        tensor.ctensor = fill(cshape, dtype, value) 
        return tensor

    @staticmethod
    def empty(shape:list, dtype:int) -> CyTensor:
        cdef Shape cshape = CyTensor.__build_shape(shape)
        cdef CyTensor tensor = CyTensor()
        tensor.ctensor = empty(cshape, dtype) 
        return tensor

    @staticmethod
    def ones(shape:list, dtype:int) -> CyTensor:
        cdef Shape cshape = CyTensor.__build_shape(shape)
        cdef CyTensor tensor = CyTensor()
        tensor.ctensor = ones(cshape, dtype) 
        return tensor

    @staticmethod
    def zeros(shape:list, dtype:int) -> CyTensor:
        cdef Shape cshape = CyTensor.__build_shape(shape)
        cdef CyTensor tensor = CyTensor()
        tensor.ctensor = zeros(cshape, dtype) 
        return tensor

    def __getitem__(self, idx:int):
        cdef CyTensor result = CyTensor()
        result.ctensor = tensor_index(&self.ctensor, idx)
        return result

    def __add__(self, other:Union[CyTensor, float]) -> CyTensor: 
        if isinstance(other, CyTensor):
            return CyTensor.add(self, other)
        else:
            return self.add_value(other)

    def __mul__(self, other:Union[CyTensor, float]) -> CyTensor:
        if isinstance(other, CyTensor):
            return CyTensor.mul(self, other)
        else:
            return self.mul_value(other)

    def __matmul__(self, other:Union[CyTensor, float]) -> CyTensor:
        return CyTensor.matmul(self, other)

    @staticmethod
    def add(t1:CyTensor, t2:CyTensor) -> CyTensor:
        cdef CyTensor result = CyTensor()
        result.ctensor = add(&t1.ctensor, &t2.ctensor)
        return result

    def add_value(self, value:float) -> CyTensor:
        add_value(&self.ctensor, value)
        return self
    
    @staticmethod
    def mul(t1:CyTensor, t2:CyTensor) -> CyTensor:
        cdef CyTensor result = CyTensor()
        result.ctensor = mul(&t1.ctensor, &t2.ctensor)
        return result

    def mul_value(self, value:float) -> CyTensor:
        mul_value(&self.ctensor, value)
        return self

    @staticmethod
    def matmul(t1:CyTensor, t2:CyTensor) -> CyTensor:
        cdef CyTensor result = CyTensor()
        result.ctensor = matmul(&t1.ctensor, &t2.ctensor)
        return result

    def __dealloc__(self):
        # Deallocate the Tensor properly when the Python object is garbage collected
        free_tensor(&self.ctensor)

    def print_tensor(self):
        print_tensor(&self.ctensor)
        print() # Extra '\n'

    def to_numpy(self) -> np.ndarray:
        # Convert C Tensor data to NumPy array for easy manipulation in Python
        cdef int32 ndim = self.ctensor.ndim
        cdef int32* shape = self.ctensor.shape.values
        cdef size_t size = self.ctensor.numel  # Number of elements in the tensor

        # Create the shape as a list to reshape the numpy array later
        shape_list = [shape[i] for i in range(ndim)]

        # Declare memoryview to hold the data
        cdef float32[:] float32_view = None
        cdef int32[:] int32_view = None
        cdef uint8[:] uint8_view = None

        # Based on the dtype, cast the void* data to the appropriate type and create a memoryview
        if self.ctensor.dtype == FLOAT32_t:
            float32_view = <float32[:size]><float32*> self.ctensor.data
            return np.array(float32_view, copy=False).reshape(shape_list)
        elif self.ctensor.dtype == INT32_t:
            int32_view = <int32[:size]><int32*> self.ctensor.data
            return np.array(int32_view, copy=False).reshape(shape_list)
        elif self.ctensor.dtype == UINT8_t:
            uint8_view = <uint8[:size]><uint8*> self.ctensor.data
            return np.array(uint8_view, copy=False).reshape(shape_list)
        else:
            raise ValueError("Unsupported tensor data type")