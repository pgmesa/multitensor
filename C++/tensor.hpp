#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <variant>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <memory>
#include <limits>
#include <iomanip>
#include <functional>

const int DECIMALS = 4;
const int VALUES_PER_LINE = 8; 

// Define aliases for types
using uint8 = uint8_t;
using int32 = int32_t;
using float32 = float;

// Enum class for data types
enum class DataType {
    UINT8,
    INT32,
    FLOAT32,
};

// Variant for TensorValue
using TensorValue = std::variant<uint8, int32, float32>;

using TensorData = std::variant<
    std::shared_ptr<uint8[]>,
    std::shared_ptr<int32[]>,
    std::shared_ptr<float32[]>
>;

// Function declarations
size_t get_dtype_size(const DataType dtype);
std::string dtype_to_str(const DataType dtype);
TensorValue cast_value(double value, const DataType dtype);

class Tensor {
public:
    TensorData data;
    size_t numel;
    std::vector<int> shape;
    int ndim;
    DataType dtype;
    std::vector<int> strides;

    // Constructors
    Tensor();
    Tensor(const TensorData& data, const size_t numel, const std::vector<int>& shape, const DataType& dtype);

    // Member functions
    template<typename T>
    T* get_data() const;

    static Tensor empty(const std::vector<int>& shape, const DataType& dtype);
    static Tensor fill(const std::vector<int>& shape, const DataType& dtype, double value);
    static Tensor ones(const std::vector<int>& shape, const DataType& dtype);
    static Tensor zeros(const std::vector<int>& shape, const DataType& dtype);

    // Operator overloads
    Tensor operator+(const Tensor& t2) const;
    Tensor operator+(const double value) const;
    Tensor& operator+=(const Tensor& t2);
    Tensor& operator+=(const double value);
    Tensor operator*(const Tensor& t2) const;
    Tensor operator*(const double value) const;
    Tensor& operator*=(const Tensor& t2);
    Tensor& operator*=(const double value);
    Tensor operator[](int index) const;

    // Matrix multiplication
    Tensor matmul(const Tensor& t2) const;
    static Tensor matmul(const Tensor& t1, const Tensor& t2);

    // String representation
    std::string to_string() const;

private:
    std::vector<int> calc_strides() const;

    template<typename T>
    Tensor element_wise_operation(const Tensor& t2, std::function<T(T, T)> op) const;

    template<typename T>
    Tensor element_wise_operation_scalar(const double value, std::function<T(T, T)> op) const;

    template<typename T>
    void element_wise_operation_in_place(const Tensor& t2, std::function<T(T, T)> op);

    template<typename T>
    void element_wise_operation_in_place_scalar(const double value, std::function<T(T, T)> op);

    template<typename T>
    static Tensor matmul_template(const Tensor& t1, const Tensor& t2);
};

// Template function declarations
template<typename T>
std::string tensor_to_string(const Tensor& tensor, int padding);

template <typename T>
std::string array_to_string(const T* array, size_t length, int padding);

#endif // TENSOR_H