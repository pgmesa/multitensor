
#include "tensor.hpp"

// Get the size of the data type
size_t get_dtype_size(const DataType dtype) {
    switch (dtype) {
        case DataType::UINT8:   return sizeof(uint8);
        case DataType::INT32:   return sizeof(int32);
        case DataType::FLOAT32: return sizeof(float32);
        default:                return 0;
    }
}

// Convert DataType to string
std::string dtype_to_str(const DataType dtype) {
    switch (dtype) {
        case DataType::UINT8:   return "uint8";
        case DataType::INT32:   return "int32";
        case DataType::FLOAT32: return "float32";
        default:                return "unknown";
    }
}

TensorValue cast_value(double value, const DataType dtype) {
    TensorValue tvalue;
    
    switch (dtype) {
        case DataType::UINT8:
            // Clamp value to the range of uint8_t (0 to 255)
            if (value > std::numeric_limits<uint8>::max()) {
                tvalue = std::numeric_limits<uint8>::max();
            } else if (value < std::numeric_limits<uint8>::min()) {
                tvalue = std::numeric_limits<uint8>::min();
            } else {
                tvalue = static_cast<uint8>(value);
            }
            break;

        case DataType::INT32:
            tvalue = static_cast<int32>(value);
            break;

        case DataType::FLOAT32:
            tvalue = static_cast<float32>(value);
            break;
        default:
            throw std::invalid_argument("Unsupported data type.");
    }
    
    return tvalue;
}

// Default constructor
Tensor::Tensor() {}

Tensor::Tensor(const TensorData& data,
       const size_t numel,
       const std::vector<int>& shape,
       const DataType& dtype) 
    : data(data), numel(numel), shape(shape), ndim(static_cast<int>(shape.size())), dtype(dtype), strides(calc_strides()) {

    size_t numel_check = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    if (numel != numel_check) {
        throw std::runtime_error("numel does not match the product of shape elements.");
    }
}

/**
 * @brief Retrieves the raw pointer from a `std::shared_ptr` that holds an array of type `T`.
 * 
 * This function extracts the `std::shared_ptr<T[]>` from a `std::variant` named `data` and 
 * returns the underlying raw pointer (`T*`). The function assumes that the `data` member 
 * is a `std::variant` which contains a `std::shared_ptr<T[]>`.
 *
 * @tparam T The type of the array elements stored in the `std::shared_ptr`.
 * @return T* A raw pointer to the underlying array of type `T`. 
 *         If the `shared_ptr` is empty, it returns `nullptr`.
 *
 * @throws std::bad_variant_access If the `data` variant does not hold a `std::shared_ptr<T[]>`.
 */
template<typename T>
T* Tensor::get_data() const {
    return std::get<std::shared_ptr<T[]>>(data).get();
}

std::vector<int> Tensor::calc_strides() const {
    std::vector<int> strides(this->shape.size());
    size_t data_dtype_size = get_dtype_size(this->dtype);
    size_t numel = 1;
    int i = static_cast<int>(this->shape.size()-1);
    for (i; i >= 0; i--) {
        strides[i] = static_cast<int>(numel * data_dtype_size);
        numel *= this->shape[i];
    }
    return strides;
}

Tensor Tensor::empty(const std::vector<int>& shape, const DataType& dtype) {
    int numel = 1;
    for (int nelem: shape) {
        numel *= nelem;
    }

    TensorData data;
    switch(dtype) {
        case DataType::UINT8:
            data = std::shared_ptr<uint8[]>(new uint8[numel]);
            break;
        case DataType::INT32:
            data = std::shared_ptr<int32[]>(new int32[numel]);
            break;
        case DataType::FLOAT32:
            data = std::shared_ptr<float32[]>(new float32[numel]);
            break;
        default:
            throw std::invalid_argument("Unsupported data type.");
    }

    Tensor tensor = Tensor(data, numel, shape, dtype);
    return tensor;
}

Tensor Tensor::fill(const std::vector<int>& shape, const DataType& dtype, double value) {
    Tensor tensor = Tensor::empty(shape, dtype);
    TensorValue cvalue = cast_value(value, dtype);

    switch (dtype) {
        case DataType::UINT8: {
            uint8* ptr_ui8 = tensor.get_data<uint8>();
            uint8 val_ui8 = std::get<uint8>(cvalue);
            for (int i = 0; i < tensor.numel; i++) {
                ptr_ui8[i] = val_ui8;
            }
            break;
        }
        case DataType::INT32: {
            int32* ptr_i32 = tensor.get_data<int32>();
            int32 val_i32 = std::get<int32>(cvalue);
            for (int i = 0; i < tensor.numel; i++) {
                ptr_i32[i] = val_i32;
            }
            break;
        }
        case DataType::FLOAT32: {
            float32* ptr_f32 = tensor.get_data<float32>();
            float32 val_f32 = std::get<float32>(cvalue);
            for (int i = 0; i < tensor.numel; i++) {
                ptr_f32[i] = val_f32;
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type.");
    }
    return tensor;
}

Tensor Tensor::ones(const std::vector<int>& shape, const DataType& dtype) {
    return fill(shape, dtype, 1.0);
}

Tensor Tensor::zeros(const std::vector<int>& shape, const DataType& dtype) {
    return fill(shape, dtype, 0.0);
}

template<typename T>
Tensor Tensor::element_wise_operation(const Tensor& t2, std::function<T(T, T)> op) const {
    if (this->dtype != t2.dtype) {
        throw std::invalid_argument("Datatypes must match.");
    }

    Tensor out = Tensor::empty(this->shape, this->dtype);
    T* t1_data = this->get_data<T>();
    T* t2_data = t2.get_data<T>();
    T* out_data = out.get_data<T>();
    for (int i=0; i < this->numel; i++) {
        out_data[i] = op(t1_data[i], t2_data[i]);
    }
    return out;
}

template<typename T>
Tensor Tensor::element_wise_operation_scalar(const double value, std::function<T(T, T)> op) const {
    Tensor out = Tensor::empty(this->shape, this->dtype);
    
    TensorValue cvalue = cast_value(value, this->dtype);
    const T val = std::get<T>(cvalue);

    T* t1_data = this->get_data<T>();
    T* out_data = out.get_data<T>();
    for (int i=0; i < this->numel; i++) {
        out_data[i] = op(t1_data[i], val);
    }
    return out;
}

template<typename T>
void Tensor::element_wise_operation_in_place(const Tensor& t2, std::function<T(T, T)> op) {
    if (this->dtype != t2.dtype) {
        throw std::invalid_argument("Datatypes must match.");
    }

    T* t1_data = this->get_data<T>();
    T* t2_data = t2.get_data<T>();
    for (int i=0; i < this->numel; i++) {
        t1_data[i] = op(t1_data[i], t2_data[i]);
    }
}

template<typename T>
void Tensor::element_wise_operation_in_place_scalar(const double value, std::function<T(T, T)> op) {
    TensorValue cvalue = cast_value(value, this->dtype);
    const T val = std::get<T>(cvalue);
    T* t1_data = this->get_data<T>();
    for (int i=0; i < this->numel; i++) {
        t1_data[i] = op(t1_data[i], val);
    }
}

Tensor Tensor::operator+(const Tensor& t2) const {
    Tensor out = std::visit([this, &t2](auto& data_ptr) -> Tensor {
        using T = std::decay_t<decltype(data_ptr.get()[0])>;
        return this->element_wise_operation<T>(t2, std::plus<T>());
    }, this->data);

    return out;
}

Tensor Tensor::operator+(const double value) const {
    Tensor out = std::visit([this, value](auto& data_ptr) -> Tensor {
        using T = std::decay_t<decltype(data_ptr.get()[0])>;
        return this->element_wise_operation_scalar<T>(value, std::plus<T>());   
    }, this->data);
    return out;
}

Tensor& Tensor::operator+=(const Tensor& t2) {
    std::visit([this, &t2](auto& data_ptr) {
        using T = std::decay_t<decltype(data_ptr.get()[0])>;
        this->element_wise_operation_in_place<T>(t2, std::plus<T>());
    }, this->data);

    return *this;
}

Tensor& Tensor::operator+=(const double value) {
    std::visit([this, value](auto& data_ptr) {
        using T = std::decay_t<decltype(data_ptr.get()[0])>; 
        this->element_wise_operation_in_place_scalar<T>(value, std::plus<T>());
    }, this->data);

    return *this;
}

Tensor Tensor::operator*(const Tensor& t2) const {
    Tensor out = std::visit([this, &t2](auto& data_ptr) -> Tensor {
        using T = std::decay_t<decltype(data_ptr.get()[0])>;
        return this->element_wise_operation<T>(t2, std::multiplies<T>());
    }, this->data);

    return out;
}

Tensor Tensor::operator*(const double value) const {
    Tensor out = std::visit([this, value](auto& data_ptr) -> Tensor {
        using T = std::decay_t<decltype(data_ptr.get()[0])>;
        return this->element_wise_operation_scalar<T>(value, std::multiplies<T>());   
    }, this->data);
    return out;
}

Tensor& Tensor::operator*=(const Tensor& t2) {
    std::visit([this, &t2](auto& data_ptr) {
        using T = std::decay_t<decltype(data_ptr.get()[0])>;
        this->element_wise_operation_in_place<T>(t2, std::multiplies<T>());
    }, this->data);

    return *this;
}

Tensor& Tensor::operator*=(const double value) {
    std::visit([this, value](auto& data_ptr) {
        using T = std::decay_t<decltype(data_ptr.get()[0])>; 
        this->element_wise_operation_in_place_scalar<T>(value, std::multiplies<T>());
    }, this->data);

    return *this;
}

Tensor Tensor::matmul(const Tensor& t2) const {
    Tensor out = std::visit([this, &t2](auto& data_ptr) -> Tensor {
        using T = std::decay_t<decltype(data_ptr.get()[0])>;
        return Tensor::matmul_template<T>(*this, t2);
    }, this->data);

    return out;
}

Tensor Tensor::matmul(const Tensor& t1, const Tensor& t2) {
    Tensor out = std::visit([&t1, &t2](auto& data_ptr) -> Tensor {
        using T = std::decay_t<decltype(data_ptr.get()[0])>;
        return Tensor::matmul_template<T>(t1, t2);
    }, t1.data);

    return out;
}

template<typename T>
Tensor Tensor::matmul_template(const Tensor& t1, const Tensor& t2) {
    if (t1.dtype != t2.dtype) {
        throw std::invalid_argument("Datatypes must match.");
    } 
    if (t1.ndim < 2 || t2.ndim < 2) {
        throw std::invalid_argument("Tensor dimensions must be greater than 1");
    }
    if (t1.shape[t1.ndim - 1] != t2.shape[t2.ndim - 2]) {
        throw std::invalid_argument("Invalid Tensor dimensions");
    }

    int ndim = t1.ndim;
    int out_t1_dim = t1.shape[ndim - 2];
    int out_t2_dim = t2.shape[ndim - 1];
    int last_t1_dim = t1.shape[ndim - 1];
    int other_t2_dim = t2.shape[ndim - 2];
    
    std::vector<int> out_shape(ndim);
    out_shape[ndim - 2] = out_t1_dim;
    out_shape[ndim - 1] = out_t2_dim;
    for (int i=0; i < ndim - 2; i++) {
        out_shape[i] = t1.shape[i];
    }
    Tensor out = Tensor::zeros(out_shape, t1.dtype);

    T* t1_data = t1.get_data<T>();
    T* t2_data = t2.get_data<T>();
    T* out_data = out.get_data<T>();

    size_t batches = t1.numel / out_t1_dim / last_t1_dim;
    for (int k=0; k < batches; k++) {
        for (int row=0; row < out_t1_dim; row++) {
            for (int j=0; j < last_t1_dim; j++) {
                for (int i=0; i < out_t2_dim; i++) {
                    int out_index = k*out_t1_dim*out_t2_dim + row*out_t2_dim + i;
                    int t1_index = k*out_t1_dim*last_t1_dim + row*last_t1_dim + j;
                    int t2_index = k*other_t2_dim*out_t2_dim + j*out_t2_dim + i;
                    
                    out_data[out_index] += t1_data[t1_index] * t2_data[t2_index];
                }
            }   
        }
    } 
    return out;
}

Tensor Tensor::operator[](int index) const {
    if (index < 0 || index >= this->shape[0]) {
        throw std::out_of_range("Index out of range");
    }

    DataType out_dtype = this->dtype;
    size_t out_numel = this->numel / this->shape[0];
    
    std::vector<int> out_shape(this->shape.begin() + 1, this->shape.end());

    size_t offset = index * out_numel;

    TensorData out_data = std::visit([offset](auto& data_ptr) -> TensorData {
        using T = std::decay_t<decltype(data_ptr.get()[0])>;
        // Create a new shared_ptr that points to the correct offset
        return std::shared_ptr<T[]>(data_ptr, data_ptr.get() + offset);
    }, this->data);

    return Tensor(out_data, out_numel, out_shape, out_dtype);
}

std::string Tensor::to_string() const {
    size_t mem_size = numel * get_dtype_size(dtype);
    std::string dtype_str = dtype_to_str(dtype);

    std::string data_str = std::visit([this](auto& data_ptr) -> std::string {
        using T = std::decay_t<decltype(data_ptr.get()[0])>; // Deduce the type of the data
        return tensor_to_string<T>(*this, 7);                // Call templated function
    }, data);
    std::string shape_str = array_to_string(shape.data(), shape.size(), 0);
    std::string strides_str = array_to_string(strides.data(), strides.size(), 0);
    
    std::ostringstream oss;
    oss << "Tensor(" << data_str << ",\n"
        << "       numel=" << numel << ", shape=" << shape_str << ", ndim=" << ndim
        << ", strides=" << strides_str << ", dtype=" << dtype_str << ", msize=" << mem_size << ")";
    return oss.str();
}

template<typename T>
std::string tensor_to_string(const Tensor& tensor, int padding) {
    size_t numel = tensor.numel;
    int ndim = tensor.ndim;
    int last_dim_size = tensor.shape[ndim - 1];
    size_t narrays = numel / last_dim_size;

    std::string buffer;

    T* data = tensor.get_data<T>();
    int dims_ended = (ndim - 1);

    for (int i = 0; i < narrays; i++) {
        if (i > 0) {
            int spaces = padding + (ndim - 1 - dims_ended);
            buffer.append(spaces, ' ');
        }
        buffer.append(dims_ended, '[');

        // Get array string
        std::string array_str = array_to_string(data, last_dim_size, 7);
        buffer.append(array_str);
        data += last_dim_size;

        // Calculate finished matrices
        dims_ended = 0;
        for (int sidx = 0; sidx < ndim - 1; sidx++) {
            if (sidx == ndim - 2) continue;
            if (((i + 1) * last_dim_size) % (tensor.strides[sidx] / sizeof(T)) == 0) {
                dims_ended += 1;
            }
        }
        buffer.append(dims_ended, ']');

        // Not the last array
        if (i < (narrays - 1)) {
            buffer.append(",\n");
            buffer.append(dims_ended, '\n');
        }
    }
    if (ndim > 1) buffer.append("]");
    return buffer;
}

template <typename T>
std::string array_to_string(const T* array, size_t length, int padding) {
    std::ostringstream oss;
    oss << "[";

    for (int i = 0; i < length; ++i) {
        // Check if T is an integral type (e.g., int, uint8_t)
        if constexpr (std::is_integral_v<T>) {
            oss << static_cast<int>(array[i]);
        } 
        // Check if T is a floating-point type (e.g., float, double)
        else if constexpr (std::is_floating_point_v<T>) {
            oss << std::fixed << std::setprecision(DECIMALS) << array[i];
        } 
        else {
            throw std::invalid_argument("Unsupported data type.");
        }

        if (i < length - 1) {
            oss << ", ";
            if ((i + 1) % VALUES_PER_LINE == 0) {
                oss << "\n";
                // Add padding spaces
                oss << std::string(padding, ' ');
            }
        }
    }
    oss << "]";

    return oss.str();
}

// Main function
int main() {
    // Creating and printing t1
    std::cout << "\n=== Creating and printing t1 ===\n\n";
    std::vector<int> shape = {3, 4, 2};
    DataType dtype = DataType::FLOAT32; 
    Tensor t1 = Tensor::fill(shape, dtype, 2.0);
    std::cout << t1.to_string();

    // Creating t1_0 (index 0 of t1) and multiplying by 2
    std::cout << "\n\n=== Creating t1_0 (index 0 of t1) and multiplying by 2 ===\n\n";
    Tensor t1_0 = t1[0];
    t1_0 *= 2;
    std::cout << t1_0.to_string();

    // Printing updated t1 (change in the first 'slice')
    std::cout << "\n\n=== Printing updated t1 (change in the first 'slice') ===\n\n";
    std::cout << t1.to_string();

    // Creating and printing t2
    std::cout << "\n\n=== Creating and printing t2 ===\n\n";
    std::vector<int> shape2 = {3, 2, 4};
    Tensor t2 = Tensor::fill(shape2, dtype, 6.0);
    std::cout << t2.to_string();

    // Performing matrix multiplication (t3 = t1 @ t2) and printing result
    std::cout << "\n\n=== Performing matrix multiplication (t3 = t1 @ t2) and printing result ===\n\n";
    Tensor t3 = Tensor::matmul(t1, t2);
    std::cout << t3.to_string();

    // Freeing allocated memory (Handled automatically by destructors in C++)
    std::cout << "\n\n=== Freeing allocated memory ===\n";
    
    // In C++, destructors handle memory cleanup automatically, no need to call free_tensor().
    std::cout << "\n=== Program completed successfully ===\n\n";
    return 0;
}