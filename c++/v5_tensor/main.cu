#include "tensor.h"
#include "backends/backend_ops.h"
#include "graph/node.h"
#include <iostream>

BackendOps *g_backend_ops = nullptr;

void test_plan() {
    std::cout << " print 1 " << std::endl;
    printAllTensors();
    Tensor *input = allocTensor({3, 2}, "input");
    Tensor *w = allocTensor({2, 2}, "w");
    Tensor *bias = allocTensor({2}, "bias");
    graph::Node *ni = graph::allocNode(input);
    graph::Node *nw = graph::allocNode(w);
    graph::Node *nb = graph::allocNode(bias);
    auto nres = ni->at(nw)->expand_add(nb);
    std::cout << " print 2 " << std::endl;
    printAllTensors();
    nres->backward();
    std::cout << " print 3 " << std::endl;
    printAllTensors();
    printAllActions();
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