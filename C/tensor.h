#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stdlib.h>

typedef uint8_t uint8;
typedef int32_t int32;
typedef float float32;

typedef enum {
    UINT8_t,
    INT32_t,
    FLOAT32_t,
} DataType;

typedef union {
    uint8 ui8;
    int32 i32;
    float32 f32;
} TensorValue;

typedef struct {
    int32 *values;
    int32 ndim;
} Shape;

typedef struct {
    void *data;
    size_t numel;
    Shape shape;
    int32 *strides;
    int ndim;
    DataType dtype;
    int is_view;
} Tensor;

// Function declarations
// 1. Tensor creation and display
Tensor fill(Shape shape, DataType dtype, double value);
Tensor empty(Shape shape, DataType dtype);
Tensor zeros(Shape shape, DataType dtype);
Tensor ones(Shape shape, DataType dtype);
void free_tensor(Tensor *tensor);
void print_tensor(const Tensor *tensor);

// 2. Tensor operations
Tensor add(Tensor *t1, Tensor *t2);
void add_value(Tensor *t1, double value);
Tensor mul(Tensor *t1, Tensor *t2);
void mul_value(Tensor *t1, double value);
Tensor matmul(Tensor *t1, Tensor *t2);
Tensor tensor_index(Tensor *tensor, int idx);

#endif