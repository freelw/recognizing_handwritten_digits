#include "tensor.h"
#include "backends/cpu/cpu_ops.h"
#include "graph/node.h"
#include "common.h"

void test_plan() {
    init_backend();
    
    printAllTensors();
    printAllActions();
    allocMemAndInitTensors();
    gDoActions();
    
    freeAllTensors();
    release_backend();
}

void test_bp() {
    init_backend();
    Tensor *input = allocTensor({1, 2}, "input");
    Tensor *w = allocTensor({3, 2}, "w");
    Tensor *bias = allocTensor({3}, "bias");
    Tensor *w1 = allocTensor({3, 3}, "w1");
    Tensor *bias1 = allocTensor({3}, "bias1");

    graph::Node *ni = graph::allocNode(input);
    graph::Node *nw = graph::allocNode(w);
    graph::Node *nb = graph::allocNode(bias);
    graph::Node *nw1 = graph::allocNode(w1);
    graph::Node *nb1 = graph::allocNode(bias1);

    Tensor *labels = allocTensor({1}, "labels", INT32);
    auto foward_res0 = ni->at(nw->transpose())
        ->expand_add(nb);//->relu();
    // auto foward_res1 = foward_res0
    //     ->at(nw1->transpose())
    //     ->expand_add(nb1);
    // auto nres = foward_res1
    //     ->CrossEntropy(labels);

    // zero_grad();
    // nres->backward();
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

    float *w1_data = static_cast<float*>(w1->get_data());
    for (int i = 0; i < w1->length(); ++i) {
        w1_data[i] = 0.1f;
    }

    float *bias1_data = static_cast<float*>(bias1->get_data());
    for (int i = 0; i < bias1->length(); ++i) {
        bias1_data[i] = 0.1f;
    }

    w_data[0] = 0.9f;
    w_data[1*w->get_shape()[1]] = -0.9f;

    w1_data[0] = 0.9f;
    w1_data[1*w1->get_shape()[1]] = -0.9f;

    // print w_data
    std::cout << "w_data: " << std::endl;
    for (int i = 0; i < w->length(); ++i) {
        std::cout << static_cast<float*>(w->get_data())[i] << " ";
    }
    std::cout << std::endl;

    // print w1_data
    // std::cout << "w1_data: " << std::endl;
    // for (int i = 0; i < w1->length(); ++i) {
    //     std::cout << static_cast<float*>(w1->get_data())[i] << " ";
    // }
    // std::cout << std::endl;

    gDoActions();

    // print forward result
    std::cout << "forward result0: " << std::endl;
    for (int i = 0; i < foward_res0->get_tensor()->length(); ++i) {
        std::cout << static_cast<float*>(foward_res0->get_tensor()->get_data())[i] << " ";
    }
    std::cout << std::endl;
    // std::cout << "forward result1: " << std::endl;
    // for (int i = 0; i < foward_res1->get_tensor()->length(); ++i) {
    //     std::cout << static_cast<float*>(foward_res1->get_tensor()->get_data())[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "loss : " << static_cast<float*>(nres->get_tensor()->get_data())[0] << std::endl;
    freeAllTensors();
    release_backend();
}

int main() {
    
    freeAllTensors();
    freeAllTensorViews();
    graph::freeAllEdges();
    graph::freeAllNodes();
    freeAllActions();
    return 0;
}