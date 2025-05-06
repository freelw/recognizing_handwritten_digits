#include "tensor.h"
#include "backends/cpu/cpu_ops.h"
#include "graph/node.h"
#include "common.h"

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
    for (int i = 0; i < res_wi_tensor->length(); ++ i) {
        if (fabs(res_wi_data[i] - res_wti_data[i]) > eps) {
            std::cerr << "Error: res_wi[" << i << "] = " << res_wi_data[i]
                      << ", res_wti[" << i << "] = " << res_wti_data[i] << std::endl;
        }
    }
    std::cout << "res_wi == res_wti_tensor " << std::endl;
    freeAllTensors();
    release_backend();
}

void test_transpose() {
    test_at();
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