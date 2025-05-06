#include "tensor.h"
#include "backends/cpu/cpu_ops.h"
#include "graph/node.h"
#include "common.h"

const std::string RED = "\033[31m";
const std::string GREEN = "\033[32m";
const std::string RESET = "\033[0m";

void test_at() {
    init_backend();
    Tensor *input = allocTensor({2, 3}, "input");
    Tensor *w = allocTensor({3, 4}, "w");
    Tensor *wt = allocTensor({4, 3}, "wt");
    graph::Node *ni = graph::allocNode(input);
    graph::Node *nw = graph::allocNode(w);
    graph::Node *nwt = graph::allocNode(wt);
    auto res_wi = ni->at(nw);
    auto res_wti = ni->at(nwt->transpose());
    // printAllTensors();
    // printAllActions();
    allocMemAndInitTensors();
    input->fill(1.0f);
    for (int i = 0; i < 3; ++ i) {
        for (int j = 0; j < 4; ++ j) {
            float *loc_w = w->location({i, j});
            float *loc_wt = wt->location({j, i});
            float v = i * 4 + j;
            *loc_w = v;
            *loc_wt = v;
        }
    }
    gDoActions();
    auto res_wi_tensor = res_wi->get_tensor();
    auto res_wti_tensor = res_wti->get_tensor();
    auto res_wi_data = static_cast<float*>(res_wi_tensor->get_data());
    auto res_wti_data = static_cast<float*>(res_wti_tensor->get_data());
    const float eps = 1e-5f;
    bool succ = true;
    for (int i = 0; i < res_wi_tensor->length(); ++ i) {
        if (fabs(res_wi_data[i] - res_wti_data[i]) > eps) {
            succ = false;
            std::cerr << RED << "Error: res_wi[" << i << "] = " << res_wi_data[i]
                      << ", res_wti[" << i << "] = " << res_wti_data[i] << RESET << std::endl;
        }
    }
    if (succ) {
        std::cout << GREEN << "test_at succ " << RESET << std::endl;
    }
    // // print res_wi shape
    // std::cout << "res_wi shape: ";
    // assert(res_wi_tensor->get_shape().size() == 2);
    // for (int i = 0; i < res_wi_tensor->get_shape().size(); ++ i) {
    //     std::cout << res_wi_tensor->get_shape()[i] << " ";
    // }
    // std::cout << std::endl;

    // // print res_wi data
    // std::cout << "res_wi data: " << std::endl;
    // for (int i = 0; i < res_wi_tensor->get_shape()[0]; ++ i) {
    //     for (int j = 0; j < res_wi_tensor->get_shape()[1]; ++ j) {
    //         std::cout << res_wi_data[i * res_wi_tensor->get_shape()[1] + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    freeAllActions();
    freeAllTensors();
    releaseTensorMem();
    release_backend();
}

void test_add() {
    init_backend();
    Tensor *input = allocTensor({3, 4}, "input");
    Tensor *w = allocTensor({3, 4}, "w");
    Tensor *wt = allocTensor({4, 3}, "wt");
    Tensor *res_wi_tensor = allocTensor({3, 4}, "res_wi");
    Tensor *res_wti_tensor = allocTensor({3, 4}, "res_wti");
    gCreateAction(
        new AddAction(input, w, res_wi_tensor)
    );
    gCreateAction(
        new AddAction(input, wt->transpose(), res_wti_tensor)
    );
    // printAllTensors();
    // printAllActions();
    allocMemAndInitTensors();
    input->fill(0.1f);
    for (int i = 0; i < 3; ++ i) {
        for (int j = 0; j < 4; ++ j) {
            float *loc_w = w->location({i, j});
            float *loc_wt = wt->location({j, i});
            float v = i * 4 + j;
            *loc_w = v;
            *loc_wt = v;
        }
    }
    gDoActions();
    auto res_wi_data = static_cast<float*>(res_wi_tensor->get_data());
    auto res_wti_data = static_cast<float*>(res_wti_tensor->get_data());
    const float eps = 1e-5f;
    bool succ = true;
    for (int i = 0; i < res_wi_tensor->length(); ++ i) {
        if (fabs(res_wi_data[i] - res_wti_data[i]) > eps) {
            succ = false;
            std::cerr << RED << "Error: res_wi[" << i << "] = " << res_wi_data[i]
                      << ", res_wti[" << i << "] = " << res_wti_data[i] << RESET << std::endl;
        }
    }
    if (succ) {
        std::cout << GREEN << "test_add succ" << RESET << std::endl;
    }

    sanitizeTensors();
    // // print res_wi shape
    // std::cout << "res_wi shape: " << res_wi_tensor->get_shape().size() << std::endl; 
    // assert(res_wi_tensor->get_shape().size() == 2);
    // for (int i = 0; i < res_wi_tensor->get_shape().size(); ++ i) {
    //     std::cout << res_wi_tensor->get_shape()[i] << " ";
    // }
    // std::cout << std::endl;

    // // print res_wi data
    // std::cout << "res_wi data: " << std::endl;
    // for (int i = 0; i < res_wi_tensor->get_shape()[0]; ++ i) {
    //     for (int j = 0; j < res_wi_tensor->get_shape()[1]; ++ j) {
    //         std::cout << res_wi_data[i * res_wi_tensor->get_shape()[1] + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    freeAllActions();
    freeAllTensors();
    releaseTensorMem();
    release_backend();
}

void test_transpose() {
    test_at();
    test_add();
}

int main() {
    test_transpose();
    freeAllTensors();
    freeAllTensorViews();
    graph::freeAllEdges();
    graph::freeAllNodes();
    freeAllActions();
    return 0;
}