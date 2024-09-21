#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


#define DECIMALS 4
#define VALUES_PER_LINE 8

size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case UINT8_t:   return sizeof(uint8);
        case INT32_t:   return sizeof(int32);
        case FLOAT32_t: return sizeof(float32);
        default:      return 0;
    }
}

const char* dtype_to_str(DataType dtype) {
    switch (dtype) {
        case UINT8_t:
            return "uint8";
        case INT32_t:
            return "int32";
        case FLOAT32_t:
            return "float32";
        default:
            return "unknown";
    }
}

void free_tensor(Tensor *tensor) {
    if (!(tensor->is_view)) {
        free(tensor->data);
    }
    tensor->data = NULL;
    free(tensor->shape.values);
    tensor->shape.values = NULL;
    free(tensor->strides);
    tensor->strides = NULL;
}

size_t estimate_arr_chars(size_t length, int padding) {
    size_t chars = length * 20 + length*2 + 2; // num_size (overestimated) + comma&spaces + brackets
    size_t extra_lines = length / VALUES_PER_LINE; 
    chars += extra_lines * padding;
    return chars; 
}

char* array_to_string(char* buffer, void *array, DataType dtype, size_t length, int padding) {
    if (buffer == NULL) {
        buffer = malloc(estimate_arr_chars(length, padding));
        if (buffer == NULL) return NULL;
        buffer[0] = '\0';
    }
    
    char *current = buffer;
    size_t remaining = estimate_arr_chars(length, padding);
    
    int written = snprintf(current, remaining, "[");
    current += written;
    remaining -= written;

    TensorValue value;
    for (int i=0; i < length; i++) {
        switch (dtype) {
            case UINT8_t:
                value.ui8 = ((uint8*)array)[i];
                written = snprintf(current, remaining, "%d", value.ui8);
                break;
            case INT32_t:
                value.i32 = ((int32*)array)[i];
                written = snprintf(current, remaining, "%d", value.i32);
                break;
            case FLOAT32_t:
                value.f32 = ((float32*)array)[i];
                written = snprintf(current, remaining, "%.*f", DECIMALS, value.f32);
                break;
        }
        current += written;
        remaining -= written;

        if (i < length - 1) {
            written = snprintf(current, remaining, ",");
            current += written;
            remaining -= written;

            if (((i + 1) % VALUES_PER_LINE) == 0) {
                written = snprintf(current, remaining, "\n");
                current += written;
                remaining -= written;

                // Padding
                for (int i=0; i < padding; i++) {
                    written = snprintf(current, remaining, " ");
                    current += written;
                    remaining -= written;
                }
            }
            written = snprintf(current, remaining, " ");
            current += written;
            remaining -= written;
        }
    }
    snprintf(current, remaining, "]");
    return buffer;
}

char* tensor_to_string(const Tensor *tensor, int padding) {
    size_t numel = tensor->numel;
    int ndim = tensor->ndim;
    int last_dim_size = tensor->shape.values[ndim - 1];
    size_t arr_size = estimate_arr_chars(last_dim_size, padding);
    size_t narrays = numel / last_dim_size;
    size_t nchars = arr_size * (narrays * (1 + padding)); 

    char *buffer = malloc(nchars);
    if (buffer == NULL) return NULL;
    buffer[0] = '\0';
    char *current = buffer;
    size_t remaining = nchars;

    size_t dtype_size = get_dtype_size(tensor->dtype);
    char *data = (char*) tensor->data;

    int dims_ended = (ndim - 1);
    int written;

    for (int i = 0; i < narrays; i++) {
        if (i > 0) {
            int spaces = padding + (ndim - 1 - dims_ended);
            for (int p = 0; p < spaces; p++) {
                written = snprintf(current, remaining, " ");
                current += written;
                remaining -= written;
            }
        }
        for (int p = 0; p < dims_ended; p++) {
            written = snprintf(current, remaining, "[");
            current += written;
            remaining -= written;
        }
        
        // Get array string and move pointers
        size_t start_len = strlen(buffer);
        array_to_string(current, data, tensor->dtype, last_dim_size, 7);
        data += dtype_size * last_dim_size;
        size_t end_len = strlen(buffer);
        current += (end_len - start_len);
        remaining -= (end_len - start_len);

        // Calculate finished matrices
        dims_ended = 0;
        for (int sidx = 0; sidx < ndim - 1; sidx++) {
            if (sidx == ndim - 2) continue;
            if (((i + 1) * last_dim_size) % (tensor->strides[sidx] / dtype_size) == 0) {
                dims_ended += 1; 
            } 
        }
        // Add brackets
        for (int bi = 0; bi < dims_ended; bi++) {
            written = snprintf(current, remaining, "]");
            current += written;
            remaining -= written;
        }
        // Not the last array
        if (i < (narrays - 1)) {
            written = snprintf(current, remaining, ",\n");
            current += written;
            remaining -= written;
            // Add blank lines in between finished matrices 
            for (int si = 0; si < dims_ended; si++) {
                written = snprintf(current, remaining, "\n");
                current += written;
                remaining -= written;
            }
        }
    }
    if (ndim > 1) {
        snprintf(current, remaining, "]");
    }
    return buffer;
}

void print_tensor(const Tensor *tensor) {
    char *data_str = tensor_to_string(tensor, 7);
    char *strides_str = array_to_string(NULL, tensor->strides, INT32_t, tensor->ndim, 0);
    char *shape_str = array_to_string(NULL, tensor->shape.values, INT32_t, tensor->ndim, 0);
    const char *dtype_str = dtype_to_str(tensor->dtype);

    size_t mem_size = tensor->numel * get_dtype_size(tensor->dtype);
    printf(
        "Tensor(%s,\n       numel=%zd, shape=%s, ndim=%d, strides=%s, dtype=%s, msize=%zd)",
        data_str, tensor->numel, shape_str, tensor->ndim, strides_str, dtype_str, mem_size
    );
    free(data_str);
    free(shape_str);
    free(strides_str);
}

Tensor empty(Shape shape, DataType dtype){
    int32 *strides = malloc(shape.ndim * get_dtype_size(INT32_t));
    int numel = 1;
    size_t data_dtype_size = get_dtype_size(dtype);

    if (strides != NULL) {
        for (int i=shape.ndim-1; i >= 0; i--) {
            strides[i] = numel * (int32)data_dtype_size;
            numel *= shape.values[i];
        }
    }
    int *data = malloc(numel * data_dtype_size);
    Tensor tensor = {data, numel, shape, strides, shape.ndim, dtype, 0};
    return tensor;
}

TensorValue cast_value(double value, DataType dtype) {
    TensorValue tvalue;
    
    switch (dtype) {
        case UINT8_t:
            // Check for range and handle overflow/underflow
            if (value > 255 || value < 0) {
                tvalue.ui8 = (value > 255) ? 255: 0;  // Handle overflow/underflow
            } else {
                tvalue.ui8 = (uint8)(int)value;  // Cast to int first to avoid fractional truncation issues
            }
            break;
        case INT32_t:
            tvalue.i32 = (int32)value;
            break;
        case FLOAT32_t:
            tvalue.f32 = (float32)value;
            break;
    }
    
    return tvalue;
}

Tensor fill(Shape shape, DataType dtype, double value) {
    TensorValue tvalue = cast_value(value, dtype);
    Tensor t = empty(shape, dtype);
    
    if (t.data != NULL) {
        for (size_t i = 0; i < t.numel; i++) {
            switch (dtype) {
                case UINT8_t:
                    ((uint8*)t.data)[i] = tvalue.ui8;
                    break;
                case INT32_t:
                    ((int32*)t.data)[i] = tvalue.i32;
                    break;
                case FLOAT32_t:
                    ((float32*)t.data)[i] = tvalue.f32;
                    break;
            }
        }
    }
    return t;
}

Tensor ones(Shape shape, DataType dtype){
    return fill(shape, dtype, 1);    
}

Tensor zeros(Shape shape, DataType dtype){
    return fill(shape, dtype, 0);    
}

Tensor add(Tensor *t1, Tensor *t2) {
    Tensor out;
    if (t1->dtype != t2->dtype) {
        printf("[!] Datatypes must match");
        return out;
    }
    out = empty(t1->shape, t1->dtype);

    for (int i=0; i < t1->numel; i++) {
        switch (t1->dtype) {
            case UINT8_t:
                ((uint8*)out.data)[i] = ((uint8*)t1->data)[i] + ((uint8*)t2->data)[i];
                break;
            case INT32_t:
                ((int32*)out.data)[i] = ((int32*)t1->data)[i] + ((int32*)t2->data)[i];
                break;
            case FLOAT32_t:
                ((float32*)out.data)[i] = ((float32*)t1->data)[i] + ((float32*)t2->data)[i];
                break;
        }
    }
    return out;
}

/*
In-place operation
*/
void add_value(Tensor *t1, double value) {
    for (int i=0; i < t1->numel; i++) {
        switch (t1->dtype) {
            case UINT8_t:
                ((uint8*)t1->data)[i] += cast_value(value, t1->dtype).ui8;
                break;
            case INT32_t:
                ((int32*)t1->data)[i] += cast_value(value, t1->dtype).i32;
                break;
            case FLOAT32_t:
                ((float32*)t1->data)[i] += cast_value(value, t1->dtype).f32;
                break;
        }
    }
}

Tensor mul(Tensor *t1, Tensor *t2) {
    Tensor out;
    if (t1->dtype != t2->dtype) {
        printf("[!] Datatypes must match");
        return out;
    }
    out = empty(t1->shape, t1->dtype);

    for (int i=0; i < t1->numel; i++) {
        switch (t1->dtype) {
            case UINT8_t:
                ((uint8*)out.data)[i] = ((uint8*)t1->data)[i] * ((uint8*)t2->data)[i];
                break;
            case INT32_t:
                ((int32*)out.data)[i] = ((int32*)t1->data)[i] * ((int32*)t2->data)[i];
                break;
            case FLOAT32_t:
                ((float32*)out.data)[i] = ((float32*)t1->data)[i] * ((float32*)t2->data)[i];
                break;
        }
    }
    return out;
}

/*
In-place operation
*/
void mul_value(Tensor *t1, double value) {
    for (int i=0; i < t1->numel; i++) {
        switch (t1->dtype) {
            case UINT8_t:
                ((uint8*)t1->data)[i] *= cast_value(value, t1->dtype).ui8;
                break;
            case INT32_t:
                ((int32*)t1->data)[i] *= cast_value(value, t1->dtype).i32;
                break;
            case FLOAT32_t:
                ((float32*)t1->data)[i] *= cast_value(value, t1->dtype).f32;
                break;
        }
    }
}


Tensor matmul(Tensor *t1, Tensor *t2) {
    Tensor out;
    if (t1->dtype != t2->dtype) {
        printf("[!] Datatypes must match");
        return out;
    }
    if (t1->ndim < 2 || t2->ndim < 2) {
        printf("[!] Tensor dimensions must be greater than 1");
        return out;
    }
    if (t1->shape.values[t1->ndim - 1] != t2->shape.values[t2->ndim - 2]) {
        printf("[!] Invalid Tensor dimensions");
        return out;
    }
    int ndim = t1->ndim;
    int32 out_t1_dim = t1->shape.values[ndim - 2];
    int32 out_t2_dim = t2->shape.values[ndim - 1];
    int32 last_t1_dim = t1->shape.values[ndim - 1];
    int32 other_t2_dim = t2->shape.values[ndim - 2];
    
    int32 *values = malloc(ndim * sizeof(int32));
    values[ndim - 2] = out_t1_dim;
    values[ndim - 1] = out_t2_dim;
    for (int i=0; i < ndim - 2; i++) {
        values[i] = t1->shape.values[i];
    }
    Shape out_shape = {values, ndim};

    out = zeros(out_shape, t1->dtype);

    size_t batches = t1->numel / out_t1_dim / last_t1_dim;
    for (int k=0; k < batches; k++) {
        for (int row=0; row < out_t1_dim; row++) {
            for (int j=0; j < last_t1_dim; j++) {
                for (int i=0; i < out_t2_dim; i++) {
                    int out_index = k*out_t1_dim*out_t2_dim + row*out_t2_dim + i;
                    int t1_index = k*out_t1_dim*last_t1_dim + row*last_t1_dim + j;
                    int t2_index = k*other_t2_dim*out_t2_dim + j*out_t2_dim + i;
                    switch (t1->dtype) {
                        case UINT8_t:
                            ((uint8*)out.data)[out_index] += ((uint8*)t1->data)[t1_index] * ((uint8*)t2->data)[t2_index];
                            break;
                        case INT32_t:
                            ((int32*)out.data)[out_index] += ((int32*)t1->data)[t1_index] * ((int32*)t2->data)[t2_index];
                            break;
                        case FLOAT32_t:
                            ((float32*)out.data)[out_index] += ((float32*)t1->data)[t1_index] * ((float32*)t2->data)[t2_index];
                            break;
                    }
                }
            }   
        }
    } 
    return out;
}


Tensor tensor_index(Tensor *tensor, int idx) {
    if (idx >= tensor->shape.values[0]) {
        printf("[!] Invalid Index\n");
        return (Tensor){0};  // Return an empty tensor
    }

    Tensor out;
    out.dtype = tensor->dtype;
    out.numel = tensor->numel / tensor->shape.values[0];
    out.ndim = tensor->ndim - 1;

    // Allocate new memory for shape and strides
    out.shape.values = malloc(out.ndim * sizeof(int32));
    out.strides = malloc(out.ndim * sizeof(int32));

    // Copy shape and strides, skipping the first dimension
    memcpy(out.shape.values, tensor->shape.values + 1, out.ndim * sizeof(int32));
    memcpy(out.strides, tensor->strides + 1, out.ndim * sizeof(int32));

    out.shape.ndim = out.ndim;

    // Calculate the correct data pointer
    out.data = (char*)tensor->data + tensor->strides[0] * idx;
    out.is_view = 1;

    return out;
}


int main() {
    printf("\n=== Creating and printing t1 ===\n\n");
    int32 values[] = {3, 4, 2};
    int ndim = sizeof(values) / sizeof(values[0]);
    Shape shape = { values, ndim };
    Tensor t1 = fill(shape, FLOAT32_t, 2.0);
    print_tensor(&t1);

    printf("\n\n=== Creating t1_0 (index 0 of t1) and multiplying by 2 ===\n\n");
    Tensor t1_0 = tensor_index(&t1, 0);
    mul_value(&t1_0, 2);
    print_tensor(&t1_0);

    printf("\n\n=== Printing updated t1 (change in the first 'slice') ===\n\n");
    print_tensor(&t1);

    printf("\n\n=== Creating and printing t2 ===\n\n");
    int32 values2[] = {3, 2, 4};
    Shape shape2 = { values2, ndim };
    Tensor t2 = fill(shape2, FLOAT32_t, 6.0);
    print_tensor(&t2);

    printf("\n\n=== Performing matrix multiplication (t3 = t1 @ t2) and printing result ===\n\n");
    Tensor t3 = matmul(&t1, &t2);
    print_tensor(&t3);

    printf("\n\n=== Freeing allocated memory ===\n");
    // Free allocated memory
    free_tensor(&t1);
    free_tensor(&t1_0); // Just to make sure views are not freed and the program doesn't crash
    free_tensor(&t2);
    free_tensor(&t3);

    printf("\n=== Program completed successfully ===\n\n");
    return 0;
}