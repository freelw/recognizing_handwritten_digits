#include "tensor.h"
#include "backends/backend_ops.h"
#include "graph/node.h"
#include <iostream>

BackendOps *g_backend_ops = nullptr;

void zero_grad() {
    gCreateAction(
        new ZeroGradAction()
    );
}

void test_plan() {
    Tensor *input = allocTensor({3, 2}, "input");
    Tensor *w = allocTensor({2, 2}, "w");
    Tensor *bias = allocTensor({2}, "bias");
    graph::Node *ni = graph::allocNode(input);
    graph::Node *nw = graph::allocNode(w);
    graph::Node *nb = graph::allocNode(bias);
    Tensor *labels = allocTensor({3}, "labels", INT8);
    auto nres = ni->at(nw)->expand_add(nb)->relu()->CrossEntropy(labels);
    zero_grad();
    nres->backward();
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