#include "tensor.h"
#include "backends/backend_ops.h"
#include "graph/node.h"
#include <iostream>

BackendOps *g_backend_ops = nullptr;


void test_plan() {
    Tensor t({2, 2});


}

int main() {
    

    test_plan();

    freeAllTensors();
    freeAllTensorViews();
    graph::freeAllEdges();
    graph::freeAllNodes();
    freeAllActions();
    return 0;
}