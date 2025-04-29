#include "tensor.h"
#include "backends/backend_ops.h"
#include "graph/node.h"
#include <iostream>

BackendOps *g_backend_ops = nullptr;

int main() {
    Tensor t({2, 2});

    std::cout << "hello tensor" << std::endl;

    freeAllTensors();
    freeAllTensorViews();
    graph::freeAllEdges();
    graph::freeAllNodes();
    return 0;
}