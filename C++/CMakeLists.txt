cmake_minimum_required(VERSION 3.12)
project(cpptensor_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization flags based on the compiler
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # Use /O2 for MSVC
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Use -O3 for GCC/Clang
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

set(CMAKE_BUILD_TYPE Release)

include_directories(${PROJECT_SOURCE_DIR}/cpptensor)

# Conda environment path
set(CONDA_ENV_PATH "C:/Users/pablo/anaconda3/envs/base-3.9")

# pybind11
set(pybind11_DIR "${CONDA_ENV_PATH}/Lib/site-packages/pybind11/share/cmake/pybind11/")
find_package(pybind11 REQUIRED)

# Add all necessary source files from the _C directory
set(TENSOR_SOURCES
    ${PROJECT_SOURCE_DIR}/cpptensor/tensor.cpp
    ${PROJECT_SOURCE_DIR}/cpptensor/dtype.cpp
    ${PROJECT_SOURCE_DIR}/cpptensor/utils.cpp
)

# Build the cpp_lib static library
add_library(tensor_cpp_lib STATIC ${TENSOR_SOURCES})

# Python bindings
pybind11_add_module(cpptensor_python tensor_bindings.cpp)

# Link cpp_lib to the Python bindings
target_link_libraries(cpptensor_python PRIVATE tensor_cpp_lib)

set_target_properties(cpptensor_python PROPERTIES
    OUTPUT_NAME cpptensor
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
