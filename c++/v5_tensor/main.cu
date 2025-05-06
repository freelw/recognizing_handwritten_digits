#include "tensor.h"
#include "backends/cpu/cpu_ops.h"
#include "graph/node.h"
#include <iostream>
#include "common.h"

void test_plan() {
    init_backend();
    Tensor *input = allocTensor({3, 2}, "input");
    Tensor *w = allocTensor({2, 2}, "w");
    Tensor *bias = allocTensor({2}, "bias");
    graph::Node *ni = graph::allocNode(input);
    graph::Node *nw = graph::allocNode(w);
    graph::Node *nb = graph::allocNode(bias);
    Tensor *labels = allocTensor({3}, "labels", INT32);
    auto nres = ni->at(nw)->expand_add(nb)->relu()->CrossEntropy(labels);
    zero_grad();
    nres->backward();
    printAllTensors();
    printAllActions();
    allocMemAndInitTensors();
    gDoActions();
    std::cout << "loss : " << static_cast<float*>(nres->get_tensor()->get_data())[0] << std::endl;
    freeAllTensors();
    release_backend();
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