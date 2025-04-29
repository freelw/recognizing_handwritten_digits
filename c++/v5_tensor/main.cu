#include "tensor.h"
#include "backends/backend_ops.h"

#include <iostream>

BackendOps *g_backend_ops = nullptr;

int main() {
    Tensor t({2, 2});

    std::cout << "hello tensor" << std::endl;
    return 0;
}