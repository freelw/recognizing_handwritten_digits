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
    gDoActions();
    std::cout << "loss : " << static_cast<float*>(nres->get_tensor()->get_data())[0] << std::endl;
    freeAllTensors();
    release_backend();
}

void test_bp() {
    init_backend();
    Tensor *input = allocTensor({1, 2}, "input");
    Tensor *w = allocTensor({2, 3}, "w");
    Tensor *bias = allocTensor({3}, "bias");
    graph::Node *ni = graph::allocNode(input);
    graph::Node *nw = graph::allocNode(w);
    graph::Node *nb = graph::allocNode(bias);
    Tensor *labels = allocTensor({1}, "labels", INT32);
    auto nres = ni->at(nw)->expand_add(nb)->relu()->CrossEntropy(labels);
    zero_grad();
    nres->backward();
    printAllTensors();
    printAllActions();
    allocMemAndInitTensors();

    float *input_data = static_cast<float*>(input->get_data());
    input_data[0] = 10.0f;
    input_data[1] = 11.0f;

    int32_t *labels_data = static_cast<int32_t*>(labels->get_data());
    labels_data[0] = 1;

    float *w_data = static_cast<float*>(w->get_data());
    for (int i = 0; i < w->length(); ++i) {
        w_data[i] = 0.1f;
    }

    float *bias_data = static_cast<float*>(bias->get_data());
    for (int i = 0; i < bias->length(); ++i) {
        bias_data[i] = 0.1f;
    }

    gDoActions();
    std::cout << "loss : " << static_cast<float*>(nres->get_tensor()->get_data())[0] << std::endl;
    freeAllTensors();
    release_backend();
}

int main() {
    // test_plan();
    test_bp();
    freeAllTensors();
    freeAllTensorViews();
    graph::freeAllEdges();
    graph::freeAllNodes();
    freeAllActions();
    return 0;
}