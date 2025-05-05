#include "tensor.h"
#include "backends/cpu/cpu_ops.h"
#include "graph/node.h"
#include <iostream>

BackendOps *g_backend_ops = nullptr;

void zero_grad() {
    gCreateAction(
        new ZeroGradAction()
    );
}

void init_backend() {
    g_backend_ops = new CPUOps();
}

void release_backend() {
    delete g_backend_ops;
    g_backend_ops = nullptr;
}

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