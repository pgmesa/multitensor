#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "cpptensor/tensor.hpp"

namespace py = pybind11;

// Factory function to create Tensor based on dtype
py::object create_tensor_ones(const std::vector<int>& shape, const DataType dt) {
    if (dt == DataType::UINT8) {
        return py::cast(Tensor<uint8>::ones(shape));
    } else if (dt == DataType::INT32) {
        return py::cast(Tensor<int32>::ones(shape));
    } else if (dt == DataType::FLOAT32) {
        return py::cast(Tensor<float32>::ones(shape));
    }

    throw std::invalid_argument("Unsupported dtype for Tensor.ones");
}

py::object create_tensor_zeros(const std::vector<int>& shape, const DataType dt) {
    if (dt == DataType::UINT8) {
        return py::cast(Tensor<uint8>::zeros(shape));
    } else if (dt == DataType::INT32) {
        return py::cast(Tensor<int32>::zeros(shape));
    } else if (dt == DataType::FLOAT32) {
        return py::cast(Tensor<float32>::zeros(shape));
    }

    throw std::invalid_argument("Unsupported dtype for Tensor.zeros");
}

py::object create_tensor_full(const std::vector<int>& shape, const DataType dt, double value) {
    if (dt == DataType::UINT8) {
        return py::cast(Tensor<uint8>::full(shape, value));
    } else if (dt == DataType::INT32) {
        return py::cast(Tensor<int32>::full(shape, value));
    } else if (dt == DataType::FLOAT32) {
        return py::cast(Tensor<float32>::full(shape, value));
    }

    throw std::invalid_argument("Unsupported dtype for Tensor.full");
}

// Templated function to bind the Tensor class
template<typename T>
void bind_tensor(py::module& m, const std::string& class_name) {
    py::class_<Tensor<T>>(m, class_name.c_str())
        .def(py::init<>())
        .def_readonly("numel", &Tensor<T>::numel)
        .def_readonly("shape", &Tensor<T>::shape)
        .def_readonly("ndim", &Tensor<T>::ndim)
        .def_readonly("dtype", &Tensor<T>::dtype)
        .def_readonly("strides", &Tensor<T>::strides)
        .def("__repr__", &Tensor<T>::to_string)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def(py::self + double())
        .def(py::self += double())
        .def(py::self * double())
        .def(py::self *= double())
        .def("view", &Tensor<T>::view)
        .def("expand", &Tensor<T>::expand)
        .def("broadcast_to", &Tensor<T>::broadcast_to)
        .def("squeeze", &Tensor<T>::squeeze)
        .def("unsqueeze", &Tensor<T>::unsqueeze)
        .def("__matmul__", static_cast<Tensor<T> (Tensor<T>::*)(const Tensor<T>&) const>(&Tensor<T>::matmul))
        .def_static("matmul", static_cast<Tensor<T> (*)(const Tensor<T>&, const Tensor<T>&)>(&Tensor<T>::matmul))
        .def_static("empty", [](const std::vector<int>& shape) {
            return Tensor<T>::empty(shape);
        })
        .def_static("full", [](const std::vector<int>& shape, double value) {
            return Tensor<T>::full(shape, value);
        })
        .def_static("ones", &Tensor<T>::ones)
        .def_static("zeros", &Tensor<T>::zeros);
}

PYBIND11_MODULE(cpptensor, m) {
    m.doc() = "pybind11 plugin for Tensor class";

    // Bind each instantiation of Tensor for the supported data types
    bind_tensor<uint8>(m, "TensorUInt8");
    bind_tensor<int32>(m, "TensorInt32");
    bind_tensor<float32>(m, "TensorFloat32");

    // Also bind the DataType enum
    py::enum_<DataType>(m, "DataType")
        .value("UINT8", DataType::UINT8)
        .value("INT32", DataType::INT32)
        .value("FLOAT32", DataType::FLOAT32);

    // Expose factory functions for Python
    m.def("ones", &create_tensor_ones, "Create a Tensor of ones", py::arg("shape"), py::arg("dtype"));
    m.def("zeros", &create_tensor_zeros, "Create a Tensor of zeros", py::arg("shape"), py::arg("dtype"));
    m.def("full", &create_tensor_full, "Create a Tensor filled with a value", py::arg("shape"), py::arg("value"), py::arg("dtype"));
}
