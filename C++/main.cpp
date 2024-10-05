
#include "cpptensor/tensor.hpp"
#include <iostream>

int main() {
    try {
        Tensor<float32> t1 = Tensor<float32>::ones({2, 3, 4});

        Tensor<float32> t2 = Tensor<float32>::full({4}, 2.0);

        std::cout << "Tensor t1:" << std::endl;
        std::cout << t1.to_string() << std::endl;

        std::cout << "Tensor t2:" << std::endl;
        std::cout << t2.to_string() << std::endl;

        Tensor<float32> t3 = t1 + t2;
        std::cout << "After element-wise addition with broadcast (t1 + t2 = t3):" << std::endl;
        std::cout << t3.to_string() << std::endl;

        Tensor<float32> t4 = t3.view({4, 3, 2});
        std::cout << "Reshaped (view) t3:" << std::endl;
        std::cout << t4.to_string() << std::endl;

        Tensor<float32> t5 = Tensor<float32>::ones({4, 2});
        std::cout << "Tensor t5:" << std::endl;
        std::cout << t5.to_string() << std::endl;

        Tensor<float32> matmul_result = Tensor<float32>::matmul(t3.view({6, 4}), t5);
        std::cout << "Matrix multiplication result (reshaped t3 @ t5):" << std::endl;
        std::cout << matmul_result.to_string() << std::endl;

        Tensor<float32> t6 = Tensor<float32>::ones({2, 1, 3, 4});
        std::cout << "Tensor t6:" << std::endl;
        std::cout << t6.to_string() << std::endl;

        Tensor<float32> t7 = Tensor<float32>::full({1, 3, 4}, 3.0);
        Tensor<float32> t8 = t6 + t7;
        std::cout << "After broadcasting and element-wise addition (t6 + t7):" << std::endl;
        std::cout << t8.to_string() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[!] Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
