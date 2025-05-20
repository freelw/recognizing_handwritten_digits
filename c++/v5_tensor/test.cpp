#include "tensor.h"
#include "backends/cpu/cpu_ops.h"
#include "graph/node.h"
#include "optimizers/parameter.h"
#include "optimizers/adam.h"
#include "model/mlp.h"
#include "module/attention.h"
#include "module/mha.h"
#include "module/embedding.h"
#include "module/posencoding.h"
#include "common.h"
#include <iomanip>
#include <cmath>
#include <unistd.h>

const std::string RED = "\033[31m";
const std::string GREEN = "\033[32m";
const std::string RESET = "\033[0m";

void init_w_wt(Tensor *w, Tensor *wt) {
    std::vector<int> w_strides = w->get_strides();
    std::vector<int> wt_strides = wt->get_strides();
    float *w_tmp_buffer = static_cast<float*>(::malloc(w->size()));
    float *wt_tmp_buffer = static_cast<float*>(::malloc(wt->size()));
    auto shape = w->get_shape();

    for (int i = 0; i < shape[0]; ++ i) {
        for (int j = 0; j < shape[1]; ++ j) {
            float *loc_w_tmp = w_tmp_buffer + i * w_strides[0] + j * w_strides[1];
            float *loc_wt_tmp = wt_tmp_buffer + j * wt_strides[0] + i * wt_strides[1];
            float v = i * shape[1] + j;
            *loc_w_tmp = v;
            *loc_wt_tmp = v;
        }
    }
    g_backend_ops->cp_to_device(
        w,
        reinterpret_cast<char*>(w_tmp_buffer),
        w->size()
    );
    g_backend_ops->cp_to_device(
        wt,
        reinterpret_cast<char*>(wt_tmp_buffer),
        wt->size()
    );
    ::free(wt_tmp_buffer);
    ::free(w_tmp_buffer);
}

void init_w_wt_for_bmm(Tensor *w, Tensor *wt) {
    std::vector<int> w_strides = w->get_strides();
    std::vector<int> wt_strides = wt->get_strides();
    float *w_tmp_buffer = static_cast<float*>(::malloc(w->size()));
    float *wt_tmp_buffer = static_cast<float*>(::malloc(wt->size()));
    auto shape = w->get_shape();

    for (int k = 0; k < shape[0]; ++ k) {    
        for (int i = 0; i < shape[1]; ++ i) {
            for (int j = 0; j < shape[2]; ++ j) {
                float *loc_w_tmp = w_tmp_buffer + k * w_strides[0] + i * w_strides[1] + j * w_strides[2];
                float *loc_wt_tmp = wt_tmp_buffer + k * wt_strides[0] + j * wt_strides[1] + i * wt_strides[2];
                float v = k*shape[1] + i * shape[2] + j;
                *loc_w_tmp = v;
                *loc_wt_tmp = v;
            }
        }
    }
    g_backend_ops->cp_to_device(
        w,
        reinterpret_cast<char*>(w_tmp_buffer),
        w->size()
    );
    g_backend_ops->cp_to_device(
        wt,
        reinterpret_cast<char*>(wt_tmp_buffer),
        wt->size()
    );
    ::free(wt_tmp_buffer);
    ::free(w_tmp_buffer);
}

bool compare_res_wi_wt_ans(
    Tensor *res_wi_tensor, Tensor *res_wti_tensor,
    float *res_ans, const std::string & test_name) {
    const float eps = 1e-5f;

    auto res_wi_data = static_cast<float*>(res_wi_tensor->get_data());
    auto res_wti_data = static_cast<float*>(res_wti_tensor->get_data());

    float *res_wi_tmp_buffer = static_cast<float*>(::malloc(res_wi_tensor->size()));
    float *res_wti_tmp_buffer = static_cast<float*>(::malloc(res_wti_tensor->size()));

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_wi_tmp_buffer),
        res_wi_tensor,
        res_wi_tensor->size()
    );

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_wti_tmp_buffer),
        res_wti_tensor,
        res_wti_tensor->size()
    );

    bool succ = true;
    if (res_ans != nullptr) {
        for (int i = 0; i < res_wi_tensor->length(); ++ i) {
            if (fabs(res_wi_tmp_buffer[i] - res_ans[i]) > eps) {
                succ = false;
                std::cerr << RED << test_name << " Error: res_wi[" << i << "] = " << res_wi_tmp_buffer[i]
                        << ", res_ans[" << i << "] = " << res_ans[i] << RESET << std::endl;
            }
        }
    }
    for (int i = 0; i < res_wi_tensor->length(); ++ i) {
        if (fabs(res_wi_tmp_buffer[i] - res_wti_tmp_buffer[i]) > eps) {
            succ = false;
            std::cerr << RED << test_name << "Error: res_wi[" << i << "] = " << res_wi_tmp_buffer[i]
                      << ", res_wti[" << i << "] = " << res_wti_tmp_buffer[i] << RESET << std::endl;
        }
    }
    if (succ) {
        std::cout << GREEN << test_name << " succ" << RESET << std::endl;
    }

    ::free(res_wti_tmp_buffer);
    ::free(res_wi_tmp_buffer);
    return succ;
}

void test_bmm() {
    construct_env();
    Tensor *input = allocTensor({1, 2, 3}, "input");
    Tensor *w = allocTensor({1, 3, 4}, "w");
    Tensor *wt = allocTensor({1, 4, 3}, "wt");
    graph::Node *ni = graph::allocNode(input);
    graph::Node *nw = graph::allocNode(w);
    graph::Node *nwt = graph::allocNode(wt);
    auto res_wi = ni->bmm(nw);
    auto res_wti = ni->bmm(nwt->transpose(1, 2));
    // printAllTensors();
    // printAllActions();
    insert_boundary_action();
    allocMemAndInitTensors();
    input->fill(1.0f);
    init_w_wt_for_bmm(w, wt);
    gDoActions();
    auto res_wi_tensor = res_wi->get_tensor();
    auto res_wti_tensor = res_wti->get_tensor();
    auto res_wi_data = static_cast<float*>(res_wi_tensor->get_data());
    auto res_wti_data = static_cast<float*>(res_wti_tensor->get_data());
    float res_ans[8] = {12,15,18,21,12,15,18,21};
    compare_res_wi_wt_ans(
        res_wi_tensor, res_wti_tensor,
        res_ans, "test_bmm"
    );
    destruct_env();
}

void test_bmm_1() {
    construct_env();
    int m = 330;
    int n = 620;
    int p = 102;
    Tensor *input = allocTensor({1, m, n}, "input");
    Tensor *w = allocTensor({1, n, p}, "w");
    Tensor *wt = allocTensor({1, p, n}, "wt");
    graph::Node *ni = graph::allocNode(input);
    graph::Node *nw = graph::allocNode(w);
    graph::Node *nwt = graph::allocNode(wt);
    auto res_wi = ni->bmm(nw);
    auto res_wti = ni->bmm(nwt->transpose(1, 2));
    // printAllTensors();
    // printAllActions();
    allocMemAndInitTensors();
    input->fill(1.0f);
    init_w_wt_for_bmm(w, wt);
    insert_boundary_action();
    gDoActions();
    auto res_wi_tensor = res_wi->get_tensor();
    auto res_wti_tensor = res_wti->get_tensor();
    auto res_wi_data = static_cast<float*>(res_wi_tensor->get_data());
    auto res_wti_data = static_cast<float*>(res_wti_tensor->get_data());
    compare_res_wi_wt_ans(
        res_wi_tensor, res_wti_tensor,
        nullptr, "test_bmm_1"
    );
    destruct_env();
}

void test_bmm_2() {
    construct_env();
    Tensor *input = allocTensor({2, 2, 3}, "input");
    Tensor *w = allocTensor({2, 3, 4}, "w");
    Tensor *wt = allocTensor({2, 4, 3}, "wt");
    graph::Node *ni = graph::allocNode(input);
    graph::Node *nw = graph::allocNode(w);
    graph::Node *nwt = graph::allocNode(wt);
    auto res_wi = ni->bmm(nw);
    auto res_wti = ni->bmm(nwt->transpose(1, 2));
    // printAllTensors();
    // printAllActions();
    insert_boundary_action();
    allocMemAndInitTensors();
    input->fill(1.0f);
    init_w_wt_for_bmm(w, wt);
    gDoActions();
    auto res_wi_tensor = res_wi->get_tensor();
    auto res_wti_tensor = res_wti->get_tensor();
    
    auto res_wi_data = static_cast<float*>(res_wi_tensor->get_data());
    auto res_wti_data = static_cast<float*>(res_wti_tensor->get_data());

    float ans[16] = {
        12, 15, 18, 21,
        12, 15, 18, 21,
        21, 24, 27, 30,
        21, 24, 27, 30
    };
    compare_res_wi_wt_ans(
        res_wi_tensor, res_wti_tensor,
        ans, "test_bmm_1"
    );
    destruct_env();
}

void test_at() {
    construct_env();
    Tensor *input = allocTensor({2, 3}, "input");
    Tensor *w = allocTensor({3, 4}, "w");
    Tensor *wt = allocTensor({4, 3}, "wt");
    graph::Node *ni = graph::allocNode(input);
    graph::Node *nw = graph::allocNode(w);
    graph::Node *nwt = graph::allocNode(wt);
    auto res_wi = ni->at(nw);
    auto res_wti = ni->at(nwt->transpose());
    insert_boundary_action();
    // printAllTensors();
    // printAllActions();
    allocMemAndInitTensors();
    input->fill(1.0f);
    init_w_wt(w, wt);
    gDoActions();
    auto res_wi_tensor = res_wi->get_tensor();
    auto res_wti_tensor = res_wti->get_tensor();
    auto res_wi_data = static_cast<float*>(res_wi_tensor->get_data());
    auto res_wti_data = static_cast<float*>(res_wti_tensor->get_data());
    float res_ans[8] = {12,15,18,21,12,15,18,21};
    compare_res_wi_wt_ans(
        res_wi_tensor, res_wti_tensor,
        res_ans, "test_at"
    );
    destruct_env();
}

void test_at_1() {
    construct_env();
    int m = 330;
    int n = 620;
    int p = 102;
    Tensor *input = allocTensor({m, n}, "input");
    Tensor *w = allocTensor({n, p}, "w");
    Tensor *wt = allocTensor({p, n}, "wt");
    graph::Node *ni = graph::allocNode(input);
    graph::Node *nw = graph::allocNode(w);
    graph::Node *nwt = graph::allocNode(wt);
    auto res_wi = ni->at(nw);
    auto res_wti = ni->at(nwt->transpose());
    insert_boundary_action();
    allocMemAndInitTensors();
    input->fill(1.0f);
    init_w_wt(w, wt);
    gDoActions();
    auto res_wi_tensor = res_wi->get_tensor();
    auto res_wti_tensor = res_wti->get_tensor();
    auto res_wi_data = static_cast<float*>(res_wi_tensor->get_data());
    auto res_wti_data = static_cast<float*>(res_wti_tensor->get_data());
    compare_res_wi_wt_ans(
        res_wi_tensor, res_wti_tensor,
        nullptr, "test_at_1"
    );
    destruct_env();
}

void test_add() {
    construct_env();
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
    insert_boundary_action();
    // printAllTensors();
    // printAllActions();
    allocMemAndInitTensors();
    input->fill(0.1f);
    init_w_wt(w, wt);
    gDoActions();

    const float eps = 1e-5f;
    float res_ans[12] = {
        0.1000000015,
        1.100000024,
        2.099999905,
        3.099999905,
        4.099999905,
        5.099999905,
        6.099999905,
        7.099999905,
        8.100000381,
        9.100000381,
        10.10000038,
        11.10000038
    };

    compare_res_wi_wt_ans(
        res_wi_tensor, res_wti_tensor,
        res_ans, "test_add"
    );
    destruct_env();
}

void test_add_eq() {
    construct_env();
    Tensor *input = allocTensor({3, 4}, "input");
    Tensor *input1 = allocTensor({3, 4}, "input1");
    Tensor *w = allocTensor({3, 4}, "w");
    Tensor *wt = allocTensor({4, 3}, "wt");
    insert_boundary_action();
    gCreateAction(
        new AddEqAction(input, w)
    );
    gCreateAction(
        new AddEqAction(input1, wt->transpose())
    );
    // printAllTensors();
    // printAllActions();
    allocMemAndInitTensors();
    input->fill(0.1f);
    input1->fill(0.1f);
    init_w_wt(w, wt);
    gDoActions();

    compare_res_wi_wt_ans(
        input, input1,
        nullptr, "test_add_eq"
    );

    destruct_env();
}

void test_expand_add() {
    construct_env();
    Tensor *bias = allocTensor({4}, "bias");
    Tensor *w = allocTensor({3, 4}, "w");
    Tensor *wt = allocTensor({4, 3}, "wt");
    Tensor *res_wi_tensor = allocTensor({3, 4}, "res_wi");
    Tensor *res_wti_tensor = allocTensor({3, 4}, "res_wti");
    gCreateAction(
        new ExpandAddAction(w, bias, res_wi_tensor)
    );
    gCreateAction(
        new ExpandAddAction(wt->transpose(), bias, res_wti_tensor)
    );
    insert_boundary_action();
    // printAllTensors();
    // printAllActions();
    allocMemAndInitTensors();
    bias->fill(0.1f);
    init_w_wt(w, wt);
    gDoActions();
    compare_res_wi_wt_ans(
        res_wi_tensor, res_wti_tensor,
        nullptr, "test_expand_add"
    );
    destruct_env();
}

void test_mul() {
    construct_env();
    Tensor *input = allocTensor({3, 4}, "input");
    Tensor *w = allocTensor({3, 4}, "w");
    Tensor *wt = allocTensor({4, 3}, "wt");
    Tensor *res_wi_tensor = allocTensor({3, 4}, "res_wi");
    Tensor *res_wti_tensor = allocTensor({3, 4}, "res_wti");
    gCreateAction(
        new MulAction(input, w, res_wi_tensor)
    );
    gCreateAction(
        new MulAction(input, wt->transpose(), res_wti_tensor)
    );
    insert_boundary_action();
    // printAllTensors();
    // printAllActions();
    allocMemAndInitTensors();
    input->fill(0.1f);
    init_w_wt(w, wt);
    gDoActions();

    compare_res_wi_wt_ans(
        res_wi_tensor, res_wti_tensor,
        nullptr, "test_mul"
    );
    destruct_env();
}

void test_mul_1() {
    construct_env();
    Tensor *input = allocTensor({1, 3, 4}, "input");
    Tensor *w = allocTensor({1, 3, 4}, "w");
    Tensor *wt = allocTensor({1, 4, 3}, "wt");
    Tensor *wtt = wt->transpose(1, 2);
    Tensor *res_wi_tensor = allocTensor({1, 3, 4}, "res_wi");
    Tensor *res_wti_tensor = allocTensor({1, 3, 4}, "res_wti");
    auto nw = graph::allocNode(w);
    nw->init_weight_for_dbg(1000.0f);
    gCreateAction(
        new MulAction(input, w, res_wi_tensor)
    );
    gCreateAction(
        new MulAction(input, wtt, res_wti_tensor)
    );
    insert_boundary_action();
    // printAllActions();
    allocMemAndInitTensors();
    input->fill(0.1f);
    float wt_data[12] = {
        0, 0.04, 0.08,
        0.01, 0.05, 0.09,
        0.02, 0.06, 0.1,
        0.03, 0.07, 0.11
    };
    g_backend_ops->cp_to_device(
        wt,
        reinterpret_cast<char*>(wt_data),
        wt->size()
    );
    gDoActions();
    compare_res_wi_wt_ans(
        res_wi_tensor, res_wti_tensor,
        nullptr, "test_mul_1"
    );
    destruct_env();
}

void test_sum() {
    construct_env();
    Tensor *w = allocTensor({3, 4}, "w");
    Tensor *wt = allocTensor({4, 3}, "wt");
    Tensor *res_wi_tensor = allocTensor({4}, "res_wi");
    Tensor *res_wti_tensor = allocTensor({4}, "res_wti");
    gCreateAction(
        new SumAction(w, res_wi_tensor, 0)
    );
    gCreateAction(
        new SumAction(wt->transpose(), res_wti_tensor, 0)
    );
    insert_boundary_action();
    // printAllTensors();
    // printAllActions();
    allocMemAndInitTensors();
    init_w_wt(w, wt);
    gDoActions();
    compare_res_wi_wt_ans(
        res_wi_tensor, res_wti_tensor,
        nullptr, "test_sum"
    );
    destruct_env();
}
void init_labels(Tensor *labels) {
    auto size = labels->size();
    int32_t *labels_tmp_buffer = static_cast<int32_t*>(::malloc(size));
    auto length = labels->length();
    for (int i = 0; i < length; ++ i) {
        labels_tmp_buffer[i] = i;
    }
    g_backend_ops->cp_to_device(
        labels,
        reinterpret_cast<char*>(labels_tmp_buffer),
        size
    );
    ::free(labels_tmp_buffer);
}

void test_cross_entropy() {
    construct_env();
    Tensor *labels = allocTensor({3}, "input", INT32);
    Tensor *w = allocTensor({3, 4}, "w");
    Tensor *wt = allocTensor({4, 3}, "wt");
    Tensor *res_wi_tensor = allocTensor({1}, "res_wi");
    Tensor *res_wti_tensor = allocTensor({1}, "res_wti");
    Tensor *maxs_wi = allocTensor({3}, "maxs_wi");
    Tensor *sums_wi = allocTensor({3}, "sums_wi");
    Tensor *maxs_wti = allocTensor({3}, "maxs_wti");
    Tensor *sums_wti = allocTensor({3}, "sums_wti");
    gCreateAction(
        new CrossEntropyAction(w, labels, maxs_wi, sums_wi, res_wi_tensor)
    );
    gCreateAction(
        new CrossEntropyAction(wt->transpose(), labels, maxs_wti, sums_wti, res_wti_tensor)
    );
    insert_boundary_action();
    // printAllTensors();
    // printAllActions();
    allocMemAndInitTensors();
    init_labels(labels);
    init_w_wt(w, wt);
    
    gDoActions();
    compare_res_wi_wt_ans(
        res_wi_tensor, res_wti_tensor,
        nullptr, "test_cross_entropy"
    );
    destruct_env();
}

void test_cross_entropy_backward() {
    construct_env();
    Tensor *labels = allocTensor({3}, "input", INT32);
    Tensor *w = allocTensor({3, 4}, "w");
    Tensor *wt = allocTensor({4, 3}, "wt");
    Tensor *res_wi_tensor = allocTensor({1}, "res_wi");
    Tensor *res_wti_tensor = allocTensor({1}, "res_wti");
    Tensor *maxs_wi = allocTensor({3}, "maxs_wi");
    Tensor *sums_wi = allocTensor({3}, "sums_wi");
    Tensor *maxs_wti = allocTensor({3}, "maxs_wti");
    Tensor *sums_wti = allocTensor({3}, "sums_wti");
    Tensor *grad_wi = allocTensor({3, 4}, "grad_wi");
    Tensor *grad_wti = allocTensor({3, 4}, "grad_wti");
    gCreateAction(
        new CrossEntropyAction(w, labels, maxs_wi, sums_wi, res_wi_tensor)
    );
    gCreateAction(
        new CrossEntropyAction(wt->transpose(), labels, maxs_wti, sums_wti, res_wti_tensor)
    );
    insert_boundary_action();
    gCreateAction(
        new CrossEntropyBackwardAction(w, labels, maxs_wi, sums_wi, grad_wi)
    );
    gCreateAction(
        new CrossEntropyBackwardAction(wt->transpose(), labels, maxs_wti, sums_wti, grad_wti)
    );
    
    // printAllTensors();
    // printAllActions();
    allocMemAndInitTensors();
    init_labels(labels);
    init_w_wt(w, wt);
    
    gDoActions();

    compare_res_wi_wt_ans(
        grad_wi, grad_wti,
        nullptr, "test_cross_entropy_backward"
    );
    destruct_env();
}

void test_bp() {
    construct_env();
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

    ni->require_grad();
    nw->require_grad();
    nb->require_grad();
    nw1->require_grad();
    nb1->require_grad();

    Tensor *labels = allocTensor({1}, "labels", INT32);
    auto foward_res0 = ni->at(nw->transpose())
        ->expand_add(nb)->relu();
    auto foward_res1 = foward_res0
        ->at(nw1->transpose())
        ->expand_add(nb1);
    auto nres = foward_res1
        ->CrossEntropy(labels);
    insert_boundary_action();
    zero_grad();
    nres->backward();
    // printAllTensors();
    // printAllActions();
    allocMemAndInitTensors();

    auto input_size = input->size();
    auto w_size = w->size();
    auto bias_size = bias->size();
    auto w1_size = w1->size();
    auto bias1_size = bias1->size();
    auto labels_size = labels->size();

    float *input_tmp_buffer = static_cast<float*>(::malloc(input_size));
    float *w_tmp_buffer = static_cast<float*>(::malloc(w_size));
    float *bias_tmp_buffer = static_cast<float*>(::malloc(bias_size));
    float *w1_tmp_buffer = static_cast<float*>(::malloc(w1_size));
    float *bias1_tmp_buffer = static_cast<float*>(::malloc(bias1_size));
    int32_t *labels_tmp_buffer = static_cast<int32_t*>(::malloc(labels_size));

    input_tmp_buffer[0] = 10.0f;
    input_tmp_buffer[1] = 11.0f;

    labels_tmp_buffer[0] = 1;

    for (int i = 0; i < w->length(); ++i) {
        w_tmp_buffer[i] = 0.1f;
    }

    for (int i = 0; i < bias->length(); ++i) {
        bias_tmp_buffer[i] = 0.1f;
    }

    for (int i = 0; i < w1->length(); ++i) {
        w1_tmp_buffer[i] = 0.1f;
    }

    for (int i = 0; i < bias1->length(); ++i) {
        bias1_tmp_buffer[i] = 0.1f;
    }

    w_tmp_buffer[0] = 0.9f;
    w_tmp_buffer[1*w->get_shape()[1]] = -0.9f;

    w1_tmp_buffer[0] = 0.9f;
    w1_tmp_buffer[1*w1->get_shape()[1]] = -0.9f;

    g_backend_ops->cp_to_device(
        input,
        reinterpret_cast<char*>(input_tmp_buffer),
        input_size
    );

    g_backend_ops->cp_to_device(
        labels,
        reinterpret_cast<char*>(labels_tmp_buffer),
        labels_size
    );

    g_backend_ops->cp_to_device(
        w,
        reinterpret_cast<char*>(w_tmp_buffer),
        w_size
    );

    g_backend_ops->cp_to_device(
        bias,
        reinterpret_cast<char*>(bias_tmp_buffer),
        bias_size
    );

    g_backend_ops->cp_to_device(
        w1,
        reinterpret_cast<char*>(w1_tmp_buffer),
        w1_size
    );

    g_backend_ops->cp_to_device(
        bias1,
        reinterpret_cast<char*>(bias1_tmp_buffer),
        bias1_size
    );

    ::free(input_tmp_buffer);
    ::free(w_tmp_buffer);
    ::free(bias_tmp_buffer);
    ::free(w1_tmp_buffer);
    ::free(bias1_tmp_buffer);
    ::free(labels_tmp_buffer);

    gDoActions();

    const float eps = 1e-5f;
    float loss = 0;
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(&loss),
        nres->get_tensor(),
        sizeof(float)
    );
    bool loss_succ = fabs(loss - 18.360287f) < eps;
    if (loss_succ) {
        std::cout << GREEN << "test_cross_entropy succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_cross_entropy failed loss : " << loss << RESET << std::endl;
    }

    auto nw_grad = nw->get_grad();
    auto nb_grad = nb->get_grad();
    auto nw1_grad = nw1->get_grad();
    auto nb1_grad = nb1->get_grad();

    auto nw_grad_size = nw_grad->size();
    auto nb_grad_size = nb_grad->size();
    auto nw1_grad_size = nw1_grad->size();
    auto nb1_grad_size = nb1_grad->size();

    auto nw_grad_shape = nw_grad->get_shape();
    auto nb_grad_shape = nb_grad->get_shape();
    auto nw1_grad_shape = nw1_grad->get_shape();
    auto nb1_grad_shape = nb1_grad->get_shape();

    auto nw_grad_strides = nw_grad->get_strides();
    auto nw1_grad_strides = nw1_grad->get_strides();

    float *nw_grad_tmp_buffer = static_cast<float*>(::malloc(nw_grad_size));
    float *nb_grad_tmp_buffer = static_cast<float*>(::malloc(nb_grad_size));
    float *nw1_grad_tmp_buffer = static_cast<float*>(::malloc(nw1_grad_size));
    float *nb1_grad_tmp_buffer = static_cast<float*>(::malloc(nb1_grad_size));

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nw_grad_tmp_buffer),
        nw_grad,
        nw_grad_size
    );

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nb_grad_tmp_buffer),
        nb_grad,
        nb_grad_size
    );

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nw1_grad_tmp_buffer),
        nw1_grad,
        nw1_grad_size
    );

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nb1_grad_tmp_buffer),
        nb1_grad,
        nb1_grad_size
    );

    bool nw_grad_succ = true;
    float nw_grad_ans[3][2] {
        17.997713,  19.797485,
        0.0000e+00,  0.0000e+00,
        -2.3890e-08, -2.6279e-08
    };

    for (int i = 0; i < nw_grad_shape[0]; ++i) {
        for (int j = 0; j < nw_grad_shape[1]; ++j) {
            auto v = nw_grad_tmp_buffer[i * nw_grad_strides[0] + j * nw_grad_strides[1]];
            if (fabs(nw_grad_ans[i][j] - v) > eps) {
                std::cerr << std::setprecision(8) << RED << "Error: nw_grad[" << i << "][" << j << "] = " << v
                          << ", nw_grad_ans[" << i << "][" << j << "] = " << nw_grad_ans[i][j] << RESET << std::endl;
                nw_grad_succ = false;
            }
        }
    }
    
    if (nw_grad_succ) {
        std::cout << GREEN << "test_cross_entropy nw_grad succ" << RESET << std::endl;
    }

    bool nb_grad_succ = true;
    float nb_grad_ans[3] = {
        1.7997713,
        0.0000e+00,
        -2.3810571e-09
    };
    
    for (int i = 0; i < nb_grad_shape[0]; ++i) {
        float v = nb_grad_tmp_buffer[i];
        if (fabs(nb_grad_ans[i] - v) > eps) {
            std::cerr << std::setprecision(8) << RED << "Error: nb_grad[" << i << "] = " << v
                      << ", nb_grad_ans[" << i << "] = " << nb_grad_ans[i] << RESET << std::endl;
            nb_grad_succ = false;
        }
    }

    if (nb_grad_succ) {
        std::cout << GREEN << "test_cross_entropy nb_grad succ" << RESET << std::endl;
    }

    float nw1_grad_ans[3][3] = {
        10.197085, 0, 2.1993711,
        -10.200001, 0, -2.1999998,
        0.002914961, 0, 0.00062871695
    };

    bool nw1_grad_succ = true;

    for (int i = 0; i < nw1_grad_shape[0]; ++i) {
        for (int j = 0; j < nw1_grad_shape[1]; ++j) {
            auto v = nw1_grad_tmp_buffer[i * nw1_grad_strides[0] + j * nw1_grad_strides[1]];
            if (fabs(nw1_grad_ans[i][j] - v) > eps) {
                std::cerr << std::setprecision(8) << RED << "Error: nw1_grad[" << i << "][" << j << "] = " << v
                          << ", nw1_grad_ans[" << i << "][" << j << "] = " << nw1_grad_ans[i][j] << RESET << std::endl;
                nw1_grad_succ = false;
            }
        }
    }
    if (nw1_grad_succ) {
        std::cout << GREEN << "test_cross_entropy nw1_grad succ" << RESET << std::endl;
    }

    float nb1_grad_ans[3] = {
        0.9997142,
        -1,
        0.00028578046
    };

    bool nb1_grad_succ = true;

    for (int i = 0; i < nb1_grad_shape[0]; ++i) {
        float v = nb1_grad_tmp_buffer[i];
        if (fabs(nb1_grad_ans[i] - v) > eps) {
            std::cerr << std::setprecision(8) << RED << "Error: nb1_grad[" << i << "] = " << v
                      << ", nb1_grad_ans[" << i << "] = " << nb1_grad_ans[i] << RESET << std::endl;
            nb1_grad_succ = false;
        }
    }

    if (nb1_grad_succ) {
        std::cout << GREEN << "test_cross_entropy nb1_grad succ" << RESET << std::endl;
    }

    ::free(nw_grad_tmp_buffer);
    ::free(nb_grad_tmp_buffer);
    ::free(nw1_grad_tmp_buffer);
    ::free(nb1_grad_tmp_buffer);

    destruct_env();
}

Tensor *calc_norm(const std::vector<Parameter*> &params) {
    Tensor *res = allocTensor({1}, "tmp_norm_res");
    std::vector<Tensor*> tensors;
    for (auto param : params) {
        tensors.push_back(param->get_grad());
    }
    gCreateAction(
        new CalcAllGradNormAction(tensors, res)
    );
    return res;
}

void test_adam() {
    construct_env();
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

    ni->require_grad();
    nw->require_grad();
    nb->require_grad();
    nw1->require_grad();
    nb1->require_grad();

    auto pnw = allocParameter(nw);
    auto pnb = allocParameter(nb);
    auto pnw1 = allocParameter(nw1);
    auto pnb1 = allocParameter(nb1);

    std::vector<Parameter*> params = {pnw, pnb, pnw1, pnb1};
    Adam adam(
        params,
        0.01f
    );

    Tensor *labels = allocTensor({1}, "labels", INT32);
    auto foward_res0 = ni->at(nw->transpose())
        ->expand_add(nb)->relu();
    auto foward_res1 = foward_res0
        ->at(nw1->transpose())
        ->expand_add(nb1);
    auto nres = foward_res1
        ->CrossEntropy(labels);
    insert_boundary_action();
    zero_grad();
    nres->backward();
    // Tensor *norm_before_clip = calc_norm(params);
    adam.clip_grad(1.0f);
    // Tensor *norm_after_clip = calc_norm(params);
    adam.step();
    // printAllTensors();
    // printAllActions();
    allocMemAndInitTensors();

    auto input_size = input->size();
    auto w_size = w->size();
    auto bias_size = bias->size();
    auto w1_size = w1->size();
    auto bias1_size = bias1->size();
    auto labels_size = labels->size();

    auto w_strides = w->get_strides();
    auto w1_strides = w1->get_strides();

    float *input_tmp_buffer = static_cast<float*>(::malloc(input_size));
    float *w_tmp_buffer = static_cast<float*>(::malloc(w_size));
    float *bias_tmp_buffer = static_cast<float*>(::malloc(bias_size));
    float *w1_tmp_buffer = static_cast<float*>(::malloc(w1_size));
    float *bias1_tmp_buffer = static_cast<float*>(::malloc(bias1_size));
    int32_t *labels_tmp_buffer = static_cast<int32_t*>(::malloc(labels_size));

    input_tmp_buffer[0] = 10.0f;
    input_tmp_buffer[1] = 11.0f;

    labels_tmp_buffer[0] = 1;

    for (int i = 0; i < w->length(); ++i) {
        w_tmp_buffer[i] = 0.1f;
    }

    for (int i = 0; i < bias->length(); ++i) {
        bias_tmp_buffer[i] = 0.1f;
    }

    for (int i = 0; i < w1->length(); ++i) {
        w1_tmp_buffer[i] = 0.1f;
    }

    for (int i = 0; i < bias1->length(); ++i) {
        bias1_tmp_buffer[i] = 0.1f;
    }

    w_tmp_buffer[0] = 0.9f;
    w_tmp_buffer[1*w->get_shape()[1]] = -0.9f;

    w1_tmp_buffer[0] = 0.9f;
    w1_tmp_buffer[1*w1->get_shape()[1]] = -0.9f;

    g_backend_ops->cp_to_device(
        input,
        reinterpret_cast<char*>(input_tmp_buffer),
        input_size
    );

    g_backend_ops->cp_to_device(
        labels,
        reinterpret_cast<char*>(labels_tmp_buffer),
        labels_size
    );

    g_backend_ops->cp_to_device(
        w,
        reinterpret_cast<char*>(w_tmp_buffer),
        w_size
    );

    g_backend_ops->cp_to_device(
        bias,
        reinterpret_cast<char*>(bias_tmp_buffer),
        bias_size
    );

    g_backend_ops->cp_to_device(
        w1,
        reinterpret_cast<char*>(w1_tmp_buffer),
        w1_size
    );

    g_backend_ops->cp_to_device(
        bias1,
        reinterpret_cast<char*>(bias1_tmp_buffer),
        bias1_size
    );

    gDoActions();

    auto nw_grad = nw->get_grad();
    auto nb_grad = nb->get_grad();
    auto nw1_grad = nw1->get_grad();
    auto nb1_grad = nb1->get_grad();

    auto nw_grad_size = nw_grad->size();
    auto nb_grad_size = nb_grad->size();
    auto nw1_grad_size = nw1_grad->size();
    auto nb1_grad_size = nb1_grad->size();

    auto nw_grad_shape = nw_grad->get_shape();
    auto nb_grad_shape = nb_grad->get_shape();
    auto nw1_grad_shape = nw1_grad->get_shape();
    auto nb1_grad_shape = nb1_grad->get_shape();

    auto nw_grad_strides = nw_grad->get_strides();
    auto nw1_grad_strides = nw1_grad->get_strides();

    float *nw_grad_tmp_buffer = static_cast<float*>(::malloc(nw_grad_size));
    float *nb_grad_tmp_buffer = static_cast<float*>(::malloc(nb_grad_size));
    float *nw1_grad_tmp_buffer = static_cast<float*>(::malloc(nw1_grad_size));
    float *nb1_grad_tmp_buffer = static_cast<float*>(::malloc(nb1_grad_size));

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nw_grad_tmp_buffer),
        nw_grad,
        nw_grad_size
    );

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nb_grad_tmp_buffer),
        nb_grad,
        nb_grad_size
    );

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nw1_grad_tmp_buffer),
        nw1_grad,
        nw1_grad_size
    );

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nb1_grad_tmp_buffer),
        nb1_grad,
        nb1_grad_size
    );

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(w_tmp_buffer),
        w,
        w_size
    );

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(bias_tmp_buffer),
        bias,
        bias_size
    );

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(w1_tmp_buffer),
        w1,
        w1_size
    );

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(bias1_tmp_buffer),
        bias1,
        bias1_size
    );

    const float eps = 1e-5f;
    bool nw_grad_succ = true;
    float nw_grad_ans[3][2] {
        0.5873974, 0.64613718,
        0, 0,
        -7.771136e-10, -8.5482493e-10,
    };
    for (int i = 0; i < nw_grad_shape[0]; ++i) {
        for (int j = 0; j < nw_grad_shape[1]; ++j) {
            auto v = nw_grad_tmp_buffer[i * nw_grad_strides[0] + j * nw_grad_strides[1]];
            if (fabs(nw_grad_ans[i][j] - v) > eps) {
                std::cerr << std::setprecision(8) << RED << "Error: nw_grad[" << i << "][" << j << "] = " << v
                          << ", nw_grad_ans[" << i << "][" << j << "] = " << nw_grad_ans[i][j] << RESET << std::endl;
                nw_grad_succ = false;
            }
        }
    }
    if (nw_grad_succ) {
        std::cout << GREEN << "test_adam clip nw_grad succ" << RESET << std::endl;
    }

    bool nb_grad_succ = true;
    float nb_grad_ans[3] = {
        0.05873974,
        0.0000e+00,
        -7.7711358e-11
    };
    
    for (int i = 0; i < nb_grad_shape[0]; ++i) {
        float v = nb_grad_tmp_buffer[i];
        if (fabs(nb_grad_ans[i] - v) > eps) {
            std::cerr << std::setprecision(8) << RED << "Error: nb_grad[" << i << "] = " << v
                      << ", nb_grad_ans[" << i << "] = " << nb_grad_ans[i] << RESET << std::endl;
            nb_grad_succ = false;
        }
    }

    if (nb_grad_succ) {
        std::cout << GREEN << "test_adam clip nb_grad succ" << RESET << std::endl;
    }

    float nw1_grad_ans[3][3] = {
        0.33280569, 0, 0.071781613,
        -0.33290085, 0, -0.071802132,
        9.5136558e-05, 0, 2.0519647e-05
    };

    bool nw1_grad_succ = true;

    for (int i = 0; i < nw1_grad_shape[0]; ++i) {
        for (int j = 0; j < nw1_grad_shape[1]; ++j) {
            auto v = nw1_grad_tmp_buffer[i * nw1_grad_strides[0] + j * nw1_grad_strides[1]];
            if (fabs(nw1_grad_ans[i][j] - v) > eps) {
                std::cerr << std::setprecision(8) << RED << "Error: nw1_grad[" << i << "][" << j << "] = " << v
                          << ", nw1_grad_ans[" << i << "][" << j << "] = " << nw1_grad_ans[i][j] << RESET << std::endl;
                nw1_grad_succ = false;
            }
        }
    }

    if (nw1_grad_succ) {
        std::cout << GREEN << "test_adam clip nw1_grad succ" << RESET << std::endl;
    }

    float nb1_grad_ans[3] = {
        0.032628007,
        -0.032637335,
        9.3271128e-06
    };

    bool nb1_grad_succ = true;
    for (int i = 0; i < nb1_grad_shape[0]; ++i) {
        float v = nb1_grad_tmp_buffer[i];
        if (fabs(nb1_grad_ans[i] - v) > eps) {
            std::cerr << std::setprecision(8) << RED << "Error: nb1_grad[" << i << "] = " << v
                      << ", nb1_grad_ans[" << i << "] = " << nb1_grad_ans[i] << RESET << std::endl;
            nb1_grad_succ = false;
        }
    }

    if (nb1_grad_succ) {
        std::cout << GREEN << "test_adam clip nb1_grad succ" << RESET << std::endl;
    }

    float w_ans[3][2] = {
        0.88999999, 0.090000004,
        -0.89999998, 0.1,
        0.10072108, 0.10078751
    };

    bool w_succ = true;
    for (int i = 0; i < w->get_shape()[0]; ++i) {
        for (int j = 0; j < w->get_shape()[1]; ++j) {
            auto v = w_tmp_buffer[i * w_strides[0] + j * w_strides[1]];
            if (fabs(w_ans[i][j] - v) > eps) {
                std::cerr << std::setprecision(8) << RED << "Error: w[" << i << "][" << j << "] = " << v
                          << ", w_ans[" << i << "][" << j << "] = " << w_ans[i][j] << RESET << std::endl;
                w_succ = false;    
            }
        }
    }

    if (w_succ) {
        std::cout << GREEN << "test_adam w succ" << RESET << std::endl;
    }

    float bias_ans[3] = {
        0.090000004, 0.1, 0.10007711
    };
    bool bias_succ = true;
    for (int i = 0; i < bias->get_shape()[0]; ++i) {
        float v = bias_tmp_buffer[i];
        if (fabs(bias_ans[i] - v) > eps) {
            std::cerr << std::setprecision(8) << RED << "Error: bias[" << i << "] = " << v
                      << ", bias_ans[" << i << "] = " << bias_ans[i] << RESET << std::endl;
            bias_succ = false;
        }
    }

    if (bias_succ) {
        std::cout << GREEN << "test_adam bias succ" << RESET << std::endl;
    }

    float w1_ans[3][3] = {
        0.88999999, 0.1, 0.090000004,
        -0.88999999, 0.1, 0.11,
        0.090001054, 0.1, 0.090004876
    };

    bool w1_succ = true;
    for (int i = 0; i < w1->get_shape()[0]; ++i) {
        for (int j = 0; j < w1->get_shape()[1]; ++j) {
            auto v = w1_tmp_buffer[i * w1_strides[0] + j * w1_strides[1]];
            if (fabs(w1_ans[i][j] - v) > eps) {
                std::cerr << std::setprecision(8) << RED << "Error: w1[" << i << "][" << j << "] = " << v
                          << ", w1_ans[" << i << "][" << j << "] = " << w1_ans[i][j] << RESET << std::endl;
                w1_succ = false;    
            }
        }
    }
    if (w1_succ) {
        std::cout << GREEN << "test_adam w1 succ" << RESET << std::endl;
    }

    float bias1_ans[3] = {
        0.090000004, 0.11, 0.09001071
    };

    bool bias1_succ = true;
    for (int i = 0; i < bias1->get_shape()[0]; ++i) {
        float v = bias1_tmp_buffer[i];
        if (fabs(bias1_ans[i] - v) > eps) {
            std::cerr << std::setprecision(8) << RED << "Error: bias1[" << i << "] = " << v
                      << ", bias1_ans[" << i << "] = " << bias1_ans[i] << RESET << std::endl;
            bias1_succ = false;
        }
    }
    if (bias1_succ) {
        std::cout << GREEN << "test_adam bias1 succ" << RESET << std::endl;
    }

    ::free(input_tmp_buffer);
    ::free(w_tmp_buffer);
    ::free(bias_tmp_buffer);
    ::free(w1_tmp_buffer);
    ::free(bias1_tmp_buffer);
    ::free(labels_tmp_buffer);
    ::free(nw_grad_tmp_buffer);
    ::free(nb_grad_tmp_buffer);
    ::free(nw1_grad_tmp_buffer);
    ::free(nb1_grad_tmp_buffer);

    destruct_env();
}

float calc_mean(Tensor *tensor) {
    float sum = 0.0f;
    auto size = tensor->size();

    auto data = static_cast<float*>(::malloc(size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(data),
        tensor,
        size
    );
    for (int i = 0; i < tensor->length(); ++i) {
        sum += data[i];
    }
    ::free(data);
    return sum / tensor->length();
}

float calc_std(Tensor *tensor) {
    float mean = calc_mean(tensor);
    float sum = 0.0f;
    auto size = tensor->size();
    auto data = static_cast<float*>(::malloc(size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(data),
        tensor,
        size
    );
    for (int i = 0; i < tensor->length(); ++i) {
        sum += (data[i] - mean) * (data[i] - mean);
    }
    ::free(data);
    return sqrt(sum / tensor->length());
}

void test_mlp() {
    construct_env();
    MLP mlp(
        784,
        {30, 10},
        0.0f
    );
    Tensor *input = allocTensor({1, 784}, "input");
    Tensor *labels = allocTensor({1}, "labels", INT32);
    auto n_input = graph::allocNode(input);
    auto res = mlp.forward(n_input)->CrossEntropy(labels);
    zero_grad();
    insert_boundary_action();
    Adam adam(
        mlp.get_parameters(),
        0.001f
    );
    res->backward();
    adam.clip_grad(1.0f);
    // adam.step(); // 这里不应该step 否则会改变参数的值
    // printAllTensors();
    allocMemAndInitTensors();
    for (int i = 0; i < 500; ++ i) {
        gDoActions();
        float loss = 0;
        g_backend_ops->cp_from_device(
            reinterpret_cast<char*>(&loss),
            res->get_tensor(),
            sizeof(float)
        );
        // std::cout << "loss : " << loss << std::endl;
    }

    auto w1_tensor = mlp.get_parameters()[0]->get_w();
    auto w2_tensor = mlp.get_parameters()[2]->get_w();

    float w1_mean = calc_mean(w1_tensor);
    float w1_std = calc_std(w1_tensor);
    float w2_mean = calc_mean(w2_tensor);
    float w2_std = calc_std(w2_tensor);

    const float eps = 0.01f;
    const float mean_ans = 0.0f;
    const float std_ans = 0.02f;
    bool w1_succ = fabs(w1_mean - mean_ans) < eps && fabs(w1_std - std_ans) < eps;
    bool w2_succ = fabs(w2_mean - mean_ans) < eps && fabs(w2_std - std_ans) < eps;

    if (w1_succ && w2_succ) {
        std::cout << GREEN << "test_mlp init weight succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_mlp init weight failed" << RESET << std::endl;
    }

    std::vector<Action*> once_actions = getOnceActions();
    bool succ = true;
    for (auto action : once_actions) {
        if (action->get_exec_times() != 1) {
            succ = false;
        }
    }
    if (succ) {
        std::cout << GREEN << "test_mlp once action succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_mlp once action failed" << RESET << std::endl;
    }
    destruct_env();
}

void test_print_tensor() {
    construct_env();
    Tensor *input = allocTensor({2, 2, 4}, "input");
    auto node = graph::allocNode(input);
    node->init_weight_gauss(0.02, 0);
    allocMemAndInitTensors();
    gDoActions();
    std::cout << *input << std::endl;
    destruct_env();
}

bool compare_res_ans(
    Tensor *res, float *ans, const std::string &name,
    float eps = 1e-5f
) {
    auto res_size = res->size();
    auto res_shape = res->get_shape();
    auto res_strides = res->get_strides();
    auto res_data = static_cast<float*>(::malloc(res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_data),
        res,
        res_size
    );
    bool succ = true;
    for (int i = 0; i < res_shape[0]; ++i) {
        for (int j = 0; j < res_shape[1]; ++j) {
            auto v = res_data[i * res_strides[0] + j * res_strides[1]];
            if (fabs(ans[i * res_shape[1] + j] - v) > eps) {
                std::cerr << std::setprecision(8) << RED << "Error: " << name
                          << "[" << i << "][" << j << "] = " << v
                          << ", ans[" << i << "][" << j << "] = " << ans[i * res_shape[1] + j] << RESET << std::endl;
                succ = false;
            }
        }
    }
    if (succ) {
        std::cout << GREEN << name << " succ" << RESET << std::endl;
    }
    ::free(res_data);
    return succ;
}

bool compare_res_ans_1d(
    Tensor *res, float *ans, const std::string &name,
    float eps = 1e-5f
) {
    auto res_size = res->size();
    auto res_length = res->length();
    auto res_data = static_cast<float*>(::malloc(res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_data),
        res,
        res_size
    );
    bool succ = true;
    for (int i = 0; i < res_length; ++i) {
        auto v = res_data[i];
        if (fabs(ans[i] - v) > eps) {
            std::cerr << std::setprecision(8) << RED << "Error: " << name
                      << "[" << i << "] = " << v
                      << ", ans[" << i << "] = " << ans[i] << RESET << std::endl;
            succ = false;
        }
    }
    if (succ) {
        std::cout << GREEN << name << " succ" << RESET << std::endl;
    }
    ::free(res_data);
    return succ;
}

bool compare_res_ans_1d_int32(
    Tensor *res, int32_t *ans, const std::string &name
) {
    auto res_size = res->size();
    auto res_length = res->length();
    auto res_data = static_cast<int32_t*>(::malloc(res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_data),
        res,
        res_size
    );
    bool succ = true;
    for (int i = 0; i < res_length; ++i) {
        auto v = res_data[i];
        if (ans[i] != v) {
            std::cerr << RED << "Error: " << name
                      << "[" << i << "] = " << v
                      << ", ans[" << i << "] = " << ans[i] << RESET << std::endl;
            succ = false;
        }
    }
    if (succ) {
        std::cout << GREEN << name << " succ" << RESET << std::endl;
    }
    ::free(res_data);
    return succ;
}

void test_reshape() {
    construct_env();

    Tensor *l = allocTensor({3, 4}, "input");
    auto n = graph::allocNode(l);
    n->init_weight_for_dbg();
    auto l_t = l->transpose();
    auto l_t_reshape = l_t->reshape({3, 4});
    auto l_r = l->reshape({4, 3});
    insert_boundary_action();
    allocMemAndInitTensors();
    // printAllActions();
    gDoActions();

    // std::cout << "l : " << std::endl << *l << std::endl;
    // std::cout << "l_t : " << std::endl << *l_t << std::endl;
    // std::cout << "l_t_reshape : " << std::endl << *l_t_reshape << std::endl;
    // std::cout << "l_r : " << std::endl << *l_r << std::endl;

    float l_ans[12] = {
        0, 1e-05, 2e-05, 3e-05,
        4e-05, 5e-05, 6e-05, 7e-05,
        8e-05, 9e-05, 0.0001, 0.00011
    };

    float l_t_ans[12] = {
        0, 4e-05, 8e-05,
        1e-05, 5e-05, 9e-05,
        2e-05, 6e-05, 0.0001,
        3e-05, 7e-05, 0.00011
    };

    float l_t_shape_ans[12] = {
        0, 4e-05, 8e-05, 1e-05,
        5e-05, 9e-05, 2e-05, 6e-05,
        0.0001, 3e-05, 7e-05, 0.00011
    };

    float l_r_ans[12] = {
        0, 1e-05, 2e-05,
        3e-05, 4e-05, 5e-05,
        6e-05, 7e-05, 8e-05,
        9e-05, 0.0001, 0.00011
    };

    bool succ = compare_res_ans(l, l_ans, "l") &&
        compare_res_ans(l_t, l_t_ans, "l_t") &&
        compare_res_ans(l_t_reshape, l_t_shape_ans, "l_t_reshape") &&
        compare_res_ans(l_r, l_r_ans, "l_r");
    
    if (succ) {
        std::cout << GREEN << "test_reshape succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_reshape failed" << RESET << std::endl;
    }
    
    destruct_env();
}

void test_reshape_1() {
    construct_env();

    Tensor *l = allocTensor({3, 4}, "input");
    auto n = graph::allocNode(l);
    n->init_weight_for_dbg();
    auto l_t = l->transpose();
    auto l_t_reshape = l_t->reshape({3, 4});
    auto l_r = l->reshape({4, 3});
    auto l_t_m_1 = l_t->reshape({-1});
    auto l_t_d3 = l_t->reshape({2, -1, 3});
    auto l_t_d3_1 = l_t->reshape({-1, 3, 2});
    insert_boundary_action();
    allocMemAndInitTensors();
    // printAllActions();
    gDoActions();

    std::string l_t_m_1_meta_ans = "Tensor(input_transpose_reshape_deep_copy)(12)";
    std::string l_t_d3_meta_ans = "Tensor(input_transpose_reshape_deep_copy)(2, 2, 3)";
    std::string l_t_d3_1_meta_ans = "Tensor(input_transpose_reshape_deep_copy)(2, 3, 2)";

    bool meta_succ = l_t_m_1->get_meta_info() == l_t_m_1_meta_ans &&
        l_t_d3->get_meta_info() == l_t_d3_meta_ans &&
        l_t_d3_1->get_meta_info() == l_t_d3_1_meta_ans;

    if (meta_succ) {
        std::cout << GREEN << "test_reshape_1 meta succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_reshape_1 meta failed" << RESET << std::endl;
    }

    float l_t_m_1_ans[12] = {
        0, 4e-05, 8e-05, 1e-05, 5e-05, 9e-05, 2e-05, 6e-05, 0.0001, 3e-05, 7e-05, 0.00011
    };
    float l_t_d3_ans[12] = {
        0, 4e-05, 8e-05, 1e-05, 5e-05, 9e-05, 2e-05, 6e-05, 0.0001, 3e-05, 7e-05, 0.00011
    };
    float l_t_d3_1_ans[12] = {
        0, 4e-05, 8e-05, 1e-05, 5e-05, 9e-05, 2e-05, 6e-05, 0.0001, 3e-05, 7e-05, 0.00011
    };

    bool l_t_m_1_succ = compare_res_ans_1d(l_t_m_1, l_t_m_1_ans, "l_t_m_1");
    bool l_t_d3_succ = compare_res_ans_1d(l_t_d3, l_t_d3_ans, "l_t_d3");
    bool l_t_d3_1_succ = compare_res_ans_1d(l_t_d3_1, l_t_d3_1_ans, "l_t_d3_1");

    bool succ = l_t_m_1_succ && l_t_d3_succ && l_t_d3_1_succ;
    if (succ) {
        std::cout << GREEN << "test_reshape_1 succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_reshape_1 failed" << RESET << std::endl;
    }
    destruct_env();
}

void test_reshape_with_cpu_base(
    int m, int n,
    float *l_ans,
    float *l_t_ans,
    float *l_t_shape_ans,
    float *l_r_ans
) {
    Tensor *l = allocTensor({m, n}, "input");
    auto node = graph::allocNode(l);
    node->init_weight_for_dbg();
    auto l_t = l->transpose();
    auto l_t_reshape = l_t->reshape({m, n});
    auto l_r = l->reshape({m, n});
    insert_boundary_action();
    allocMemAndInitTensors();
    gDoActions();

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(l_ans),
        l,
        l->size()
    );

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(l_t_ans),
        l_t,
        l_t->size()
    );

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(l_t_shape_ans),
        l_t_reshape,
        l_t_reshape->size()
    );

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(l_r_ans),
        l_r,
        l_r->size()
    );
}

bool compare_ans1_ans2(
    float *ans1, float *ans2, int size, float eps = 1e-5f
) {
    for (int i = 0; i < size; ++i) {
        if (fabs(ans1[i] - ans2[i]) > eps) {
            std::cerr << std::setprecision(8) << RED << "Error: ans1[" << i << "] = " << ans1[i]
                      << ", ans2[" << i << "] = " << ans2[i] << RESET << std::endl;
            return false;
        }
    }
    return true;
}

bool compare_ans1_ans2_int32(
    int32_t *ans1, int32_t *ans2, int size
) {
    for (int i = 0; i < size; ++i) {
        if (ans1[i] != ans2[i]) {
            std::cerr << RED << "Error: ans1[" << i << "] = " << ans1[i]
                      << ", ans2[" << i << "] = " << ans2[i] << RESET << std::endl;
            return false;
        }
    }
    return true;
}


void test_reshape_with_cpu() {

    construct_env();
    int m = 300;
    int n = 400;
    int size = m * n * sizeof(float);

    float *l_ans_cpu = static_cast<float*>(::malloc(size));
    float *l_t_ans_cpu = static_cast<float*>(::malloc(size));
    float *l_t_shape_ans_cpu = static_cast<float*>(::malloc(size));
    float *l_r_ans_cpu = static_cast<float*>(::malloc(size));

    float *l_ans_gpu = static_cast<float*>(::malloc(size));
    float *l_t_ans_gpu = static_cast<float*>(::malloc(size));
    float *l_t_shape_ans_gpu = static_cast<float*>(::malloc(size));
    float *l_r_ans_gpu = static_cast<float*>(::malloc(size));

    use_gpu(false);
    test_reshape_with_cpu_base(
        m,
        n,
        l_ans_cpu,
        l_t_ans_cpu,
        l_t_shape_ans_cpu,
        l_r_ans_cpu
    );
    destruct_env();

    use_gpu(true);
    construct_env();
    test_reshape_with_cpu_base(
        m,
        n,
        l_ans_gpu,
        l_t_ans_gpu,
        l_t_shape_ans_gpu,
        l_r_ans_gpu
    );
    destruct_env();

    bool l_succ = compare_ans1_ans2(l_ans_cpu, l_ans_gpu, m * n);
    bool l_t_succ = compare_ans1_ans2(l_t_ans_cpu, l_t_ans_gpu, m * n);
    bool l_t_shape_succ = compare_ans1_ans2(l_t_shape_ans_cpu, l_t_shape_ans_gpu, m * n);
    bool l_r_succ = compare_ans1_ans2(l_r_ans_cpu, l_r_ans_gpu, m * n);

    if (!l_succ) {
        std::cerr << RED << "test_test_reshape_with_cpu l failed" << RESET << std::endl;
    }
    if (!l_t_succ) {
        std::cerr << RED << "test_test_reshape_with_cpu l_t failed" << RESET << std::endl;
    }
    if (!l_t_shape_succ) {
        std::cerr << RED << "test_test_reshape_with_cpu l_t_shape failed" << RESET << std::endl;
    }
    if (!l_r_succ) {
        std::cerr << RED << "test_test_reshape_with_cpu l_r failed" << RESET << std::endl;
    }

    bool succ = l_succ && l_t_succ && l_t_shape_succ && l_r_succ;

    if (succ) {
        std::cout << GREEN << "test_test_reshape_with_cpu succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_test_reshape_with_cpu failed" << RESET << std::endl;
    }

    ::free(l_ans_cpu);
    ::free(l_t_ans_cpu);
    ::free(l_t_shape_ans_cpu);
    ::free(l_r_ans_cpu);
    ::free(l_ans_gpu);
    ::free(l_t_ans_gpu);
    ::free(l_t_shape_ans_gpu);
    ::free(l_r_ans_gpu);
}

void test_reshape_bp() {
    construct_env();
    Tensor *input = allocTensor({5, 4}, "input");
    Tensor *w = allocTensor({3, 2}, "w");
    Tensor *bias = allocTensor({3}, "bias");
    Tensor *w1 = allocTensor({3, 3}, "w1");
    Tensor *bias1 = allocTensor({3}, "bias1");

    graph::Node *ni = graph::allocNode(input);
    ni->require_grad();
    ni->init_weight_for_dbg();
    graph::Node *ni_t = ni->transpose();
    graph::Node *ni_t_r = ni_t->reshape({-1, 2});
    graph::Node *nw = graph::allocNode(w);
    graph::Node *nb = graph::allocNode(bias);
    graph::Node *nw1 = graph::allocNode(w1);
    graph::Node *nb1 = graph::allocNode(bias1);
    
    nw->require_grad();
    nb->require_grad();
    nw1->require_grad();
    nb1->require_grad();

    Tensor *labels = allocTensor({10}, "labels", INT32);
    auto foward_res0 = ni_t_r->at(nw->transpose())
        ->expand_add(nb)->relu();
    auto foward_res1 = foward_res0
        ->at(nw1->transpose())
        ->expand_add(nb1);
    auto nres = foward_res1
        ->CrossEntropy(labels);
    insert_boundary_action();
    zero_grad();
    nres->backward();
    // printAllActions();
    allocMemAndInitTensors();

    auto input_size = input->size();
    auto w_size = w->size();
    auto bias_size = bias->size();
    auto w1_size = w1->size();
    auto bias1_size = bias1->size();
    auto labels_size = labels->size();

    float *input_tmp_buffer = static_cast<float*>(::malloc(input_size));
    float *w_tmp_buffer = static_cast<float*>(::malloc(w_size));
    float *bias_tmp_buffer = static_cast<float*>(::malloc(bias_size));
    float *w1_tmp_buffer = static_cast<float*>(::malloc(w1_size));
    float *bias1_tmp_buffer = static_cast<float*>(::malloc(bias1_size));
    int32_t *labels_tmp_buffer = static_cast<int32_t*>(::malloc(labels_size));

    for (int i = 0; i < 10; ++i) {
        labels_tmp_buffer[i] = 1;
    }

    for (int i = 0; i < w->length(); ++i) {
        w_tmp_buffer[i] = 0.1f;
    }

    for (int i = 0; i < bias->length(); ++i) {
        bias_tmp_buffer[i] = 0.1f;
    }

    for (int i = 0; i < w1->length(); ++i) {
        w1_tmp_buffer[i] = 0.1f;
    }

    for (int i = 0; i < bias1->length(); ++i) {
        bias1_tmp_buffer[i] = 0.1f;
    }

    w_tmp_buffer[0] = 0.9f;
    w_tmp_buffer[1*w->get_shape()[1]] = -0.9f;

    w1_tmp_buffer[0] = 0.9f;
    w1_tmp_buffer[1*w1->get_shape()[1]] = -0.9f;

    g_backend_ops->cp_to_device(
        input,
        reinterpret_cast<char*>(input_tmp_buffer),
        input_size
    );

    g_backend_ops->cp_to_device(
        labels,
        reinterpret_cast<char*>(labels_tmp_buffer),
        labels_size
    );

    g_backend_ops->cp_to_device(
        w,
        reinterpret_cast<char*>(w_tmp_buffer),
        w_size
    );

    g_backend_ops->cp_to_device(
        bias,
        reinterpret_cast<char*>(bias_tmp_buffer),
        bias_size
    );

    g_backend_ops->cp_to_device(
        w1,
        reinterpret_cast<char*>(w1_tmp_buffer),
        w1_size
    );

    g_backend_ops->cp_to_device(
        bias1,
        reinterpret_cast<char*>(bias1_tmp_buffer),
        bias1_size
    );

    ::free(input_tmp_buffer);
    ::free(w_tmp_buffer);
    ::free(bias_tmp_buffer);
    ::free(w1_tmp_buffer);
    ::free(bias1_tmp_buffer);
    ::free(labels_tmp_buffer);

    gDoActions();

    const float eps = 1e-4f;
    float loss = 0;
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(&loss),
        nres->get_tensor(),
        sizeof(float)
    );
    loss /= 10;
    if (fabs(loss - 1.19474f) > eps) {
        std::cerr << RED << "Error: loss = " << loss << ", ans = 0.1" << RESET << std::endl;
    } else {
        std::cout << GREEN << "test_reshape_bp loss succ" << RESET << std::endl;
    }

    auto ni_grad = ni->get_grad();
    auto ni_grad_size = ni_grad->size();
    auto ni_grad_shape = ni_grad->get_shape();
    auto ni_grad_strides = ni_grad->get_strides();
    float *ni_grad_tmp_buffer = static_cast<float*>(::malloc(ni_grad_size));

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(ni_grad_tmp_buffer),
        ni_grad,
        ni_grad_size
    );

    float ni_grad_ans[5][4] = {
        0.0888, 0.0099, 0.0889, 0.0099,
        0.0099, 0.0889, 0.0099, 0.0889,
        0.0889, 0.0099, 0.0889, 0.0099,
        0.0099, 0.0889, 0.0099, 0.0889,
        0.0889, 0.0099, 0.0889, 0.0099
    };

    bool ni_grad_succ = true;

    for (int i = 0; i < ni_grad_shape[0]; ++i) {
        for (int j = 0; j < ni_grad_shape[1]; ++j) {
            auto v = ni_grad_tmp_buffer[i * ni_grad_strides[0] + j * ni_grad_strides[1]];
            if (fabs(ni_grad_ans[i][j] - v) > eps) {
                std::cerr << std::setprecision(8) << RED << "Error: ni_grad[" << i << "][" << j << "] = " << v
                          << ", ni_grad_ans[" << i << "][" << j << "] = " << ni_grad_ans[i][j] << RESET << std::endl;
                ni_grad_succ = false;
            }
        }
    }
    if (ni_grad_succ) {
        std::cout << GREEN << "test_reshape_bp ni_grad succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_reshape_bp ni_grad failed" << RESET << std::endl;
    }

    ::free(ni_grad_tmp_buffer);
    destruct_env();
}

void test_reshape_bp_1() {

    construct_env();
    Tensor *input = allocTensor({5, 4}, "input");
    Tensor *w = allocTensor({3, 2}, "w");
    Tensor *bias = allocTensor({3}, "bias");
    Tensor *w1 = allocTensor({3, 3}, "w1");
    Tensor *bias1 = allocTensor({3}, "bias1");

    graph::Node *ni = graph::allocNode(input);
    ni->require_grad();
    ni->init_weight_for_dbg();
    graph::Node *ni_t_r = ni->reshape({-1, 2});
    graph::Node *nw = graph::allocNode(w);
    graph::Node *nb = graph::allocNode(bias);
    graph::Node *nw1 = graph::allocNode(w1);
    graph::Node *nb1 = graph::allocNode(bias1);
    
    nw->require_grad();
    nb->require_grad();
    nw1->require_grad();
    nb1->require_grad();

    Tensor *labels = allocTensor({10}, "labels", INT32);
    auto foward_res0 = ni_t_r->at(nw->transpose())
        ->expand_add(nb)->relu();
    auto foward_res1 = foward_res0
        ->at(nw1->transpose())
        ->expand_add(nb1);
    auto nres = foward_res1
        ->CrossEntropy(labels);
    insert_boundary_action();
    zero_grad();
    nres->backward();
    // printAllActions();
    allocMemAndInitTensors();

    auto input_size = input->size();
    auto w_size = w->size();
    auto bias_size = bias->size();
    auto w1_size = w1->size();
    auto bias1_size = bias1->size();
    auto labels_size = labels->size();

    float *input_tmp_buffer = static_cast<float*>(::malloc(input_size));
    float *w_tmp_buffer = static_cast<float*>(::malloc(w_size));
    float *bias_tmp_buffer = static_cast<float*>(::malloc(bias_size));
    float *w1_tmp_buffer = static_cast<float*>(::malloc(w1_size));
    float *bias1_tmp_buffer = static_cast<float*>(::malloc(bias1_size));
    int32_t *labels_tmp_buffer = static_cast<int32_t*>(::malloc(labels_size));

    for (int i = 0; i < 10; ++i) {
        labels_tmp_buffer[i] = 1;
    }

    for (int i = 0; i < w->length(); ++i) {
        w_tmp_buffer[i] = 0.1f;
    }

    for (int i = 0; i < bias->length(); ++i) {
        bias_tmp_buffer[i] = 0.1f;
    }

    for (int i = 0; i < w1->length(); ++i) {
        w1_tmp_buffer[i] = 0.1f;
    }

    for (int i = 0; i < bias1->length(); ++i) {
        bias1_tmp_buffer[i] = 0.1f;
    }

    w_tmp_buffer[0] = 0.9f;
    w_tmp_buffer[1*w->get_shape()[1]] = -0.9f;

    w1_tmp_buffer[0] = 0.9f;
    w1_tmp_buffer[1*w1->get_shape()[1]] = -0.9f;

    g_backend_ops->cp_to_device(
        input,
        reinterpret_cast<char*>(input_tmp_buffer),
        input_size
    );

    g_backend_ops->cp_to_device(
        labels,
        reinterpret_cast<char*>(labels_tmp_buffer),
        labels_size
    );

    g_backend_ops->cp_to_device(
        w,
        reinterpret_cast<char*>(w_tmp_buffer),
        w_size
    );

    g_backend_ops->cp_to_device(
        bias,
        reinterpret_cast<char*>(bias_tmp_buffer),
        bias_size
    );

    g_backend_ops->cp_to_device(
        w1,
        reinterpret_cast<char*>(w1_tmp_buffer),
        w1_size
    );

    g_backend_ops->cp_to_device(
        bias1,
        reinterpret_cast<char*>(bias1_tmp_buffer),
        bias1_size
    );

    ::free(input_tmp_buffer);
    ::free(w_tmp_buffer);
    ::free(bias_tmp_buffer);
    ::free(w1_tmp_buffer);
    ::free(bias1_tmp_buffer);
    ::free(labels_tmp_buffer);

    gDoActions();

    const float eps = 1e-4f;
    float loss = 0;
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(&loss),
        nres->get_tensor(),
        sizeof(float)
    );
    loss /= 10;
    if (fabs(loss - 1.1947f) > eps) {
        std::cerr << RED << "Error: loss = " << loss << ", ans = 0.1" << RESET << std::endl;
    } else {
        std::cout << GREEN << "test_reshape_bp loss succ" << RESET << std::endl;
    }

    auto ni_grad = ni->get_grad();
    auto ni_grad_size = ni_grad->size();
    auto ni_grad_shape = ni_grad->get_shape();
    auto ni_grad_strides = ni_grad->get_strides();
    float *ni_grad_tmp_buffer = static_cast<float*>(::malloc(ni_grad_size));

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(ni_grad_tmp_buffer),
        ni_grad,
        ni_grad_size
    );

    float ni_grad_ans[5][4] = {
        0.0888, 0.0099, 0.0889, 0.0099,
        0.0889, 0.0099, 0.0889, 0.0099,
        0.0889, 0.0099, 0.0889, 0.0099,
        0.0889, 0.0099, 0.0889, 0.0099,
        0.0889, 0.0099, 0.0889, 0.0099
    };

    bool ni_grad_succ = true;

    for (int i = 0; i < ni_grad_shape[0]; ++i) {
        for (int j = 0; j < ni_grad_shape[1]; ++j) {
            auto v = ni_grad_tmp_buffer[i * ni_grad_strides[0] + j * ni_grad_strides[1]];
            if (fabs(ni_grad_ans[i][j] - v) > eps) {
                std::cerr << std::setprecision(8) << RED << "Error: ni_grad[" << i << "][" << j << "] = " << v
                          << ", ni_grad_ans[" << i << "][" << j << "] = " << ni_grad_ans[i][j] << RESET << std::endl;
                ni_grad_succ = false;
            }
        }
    }
    if (ni_grad_succ) {
        std::cout << GREEN << "test_reshape_bp ni_grad succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_reshape_bp ni_grad failed" << RESET << std::endl;
    }

    ::free(ni_grad_tmp_buffer);
    destruct_env();
}


void test_contiguous() {
    construct_env();
    Tensor *input = allocTensor({2, 2, 4}, "input");
    auto t_input = input->transpose();
    bool succ =
        input->is_contiguous() && !t_input->is_contiguous() && input->is_shared_with(t_input);
    if (succ) {
        std::cout << GREEN << "test_contiguous succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_contiguous failed" << RESET << std::endl;
    }
    insert_boundary_action();
    allocMemAndInitTensors();
    gDoActions();
    destruct_env();
}

void test_repeat_interleave() {
    construct_env();
    Tensor *input = allocTensor({3}, "input", INT32);
    auto node = graph::allocNode(input);
    node->init_weight_for_dbg();
    auto res = input->repeat_interleave(2);
    insert_boundary_action();
    // printAllActions();
    allocMemAndInitTensors();
    gDoActions();
    int32_t res_ans[6] = {
        0, 0, 1, 1, 2, 2
    };
    bool succ = compare_res_ans_1d_int32(res, res_ans, "res");
    if (succ) {
        std::cout << GREEN << "test_repeat_interleave succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_repeat_interleave failed" << RESET << std::endl;
    }
    destruct_env();
}

void test_mask() {
    construct_env();
    int m = 3;
    int n = 4;
    int k = 5;
    Tensor *input = allocTensor({m, n, k}, "input");
    auto ni = graph::allocNode(input);
    ni->init_weight_for_dbg();
    Tensor *mask = allocTensor({m}, "mask", INT32);
    auto nm = graph::allocNode(mask);
    nm->init_weight_for_dbg();
    auto res = input->reshape({-1, k})->sequence_mask(mask->repeat_interleave(n), 0.1f);
    insert_boundary_action();
    allocMemAndInitTensors();
    gDoActions();
    float ans[60] = {
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.0002, 0.1, 0.1, 0.1, 0.1,
        0.00025, 0.1, 0.1, 0.1, 0.1,
        0.0003, 0.1, 0.1, 0.1, 0.1,
        0.00035, 0.1, 0.1, 0.1, 0.1,
        0.0004, 0.00041, 0.1, 0.1, 0.1,
        0.00045, 0.00046, 0.1, 0.1, 0.1,
        0.0005, 0.00051, 0.1, 0.1, 0.1,
        0.00055, 0.00056, 0.1, 0.1, 0.1
    };
    bool succ = compare_res_ans(res, ans, "res");
    if (succ) {
        std::cout << GREEN << "test_mask succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_mask failed" << RESET << std::endl;
    }
    destruct_env();
}

void test_mask_1() {
    construct_env();
    int m = 3;
    int n = 4;
    int k = 13;
    Tensor *input = allocTensor({m, n, k}, "input");
    auto ni = graph::allocNode(input);
    ni->init_weight_for_dbg();
    Tensor *mask = allocTensor({m*n}, "mask", INT32);
    auto nm = graph::allocNode(mask);
    nm->init_weight_for_dbg();
    auto res = input->reshape({-1, k})->sequence_mask(mask, 0.1f);
    insert_boundary_action();
    allocMemAndInitTensors();
    gDoActions();
    float ans[156] = {
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.00013, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.00026, 0.00027, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.00039, 0.0004, 0.00041, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.00052, 0.00053, 0.00054, 0.00055, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.00065, 0.00066, 0.00067, 0.00068, 0.00069, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.00078, 0.00079, 0.0008, 0.00081, 0.00082, 0.00083, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.00091, 0.00092, 0.00093, 0.00094, 0.00095, 0.00096, 0.00097, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.00104, 0.00105, 0.00106, 0.00107, 0.00108, 0.00109, 0.0011, 0.00111, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.00117, 0.00118, 0.00119, 0.0012, 0.00121, 0.00122, 0.00123, 0.00124, 0.00125, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.00143, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
    };
    bool succ = compare_res_ans(res, ans, "res");
    if (succ) {
        std::cout << GREEN << "test_mask_1 succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_mask_1 failed" << RESET << std::endl;
    }
    destruct_env();
}

void test_softmax() {
    construct_env();
    Tensor *input = allocTensor({1, 3, 4}, "input");
    auto ni = graph::allocNode(input);
    ni->init_weight_for_dbg(10000.0f);
    auto res = input->softmax();
    insert_boundary_action();
    allocMemAndInitTensors();
    gDoActions();
    float ans[12] = {
        0.2138, 0.2363, 0.2612, 0.2887,
        0.2138, 0.2363, 0.2612, 0.2887,
        0.2138, 0.2363, 0.2612, 0.2887
    };
    bool succ_res = compare_res_ans_1d(res, ans, "res", 1e-4);
    if (!succ_res) {
        std::cout << RED << "test_softmax res failed" << RESET << std::endl;
    }

    bool succ = succ_res;
    if (succ) {
        std::cout << GREEN << "test_softmax succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_softmax failed" << RESET << std::endl;
    }    
    destruct_env();
}

void test_masked_softmax() {
    construct_env();
    Tensor *input = allocTensor({2, 2, 4}, "input");
    auto ni = graph::allocNode(input);
    ni->init_weight_for_dbg(10000.0f);
    Tensor *valid_lens = allocTensor({2}, "mask", INT32);
    auto res = ni->masked_softmax(valid_lens);
    insert_boundary_action();
    allocMemAndInitTensors();
    int valid_lens_buffer[2] = {2, 3};
    g_backend_ops->cp_to_device(
        valid_lens,
        reinterpret_cast<char*>(valid_lens_buffer),
        2 * sizeof(int32_t)
    );
    gDoActions();
    float ans[16] = {
        0.475021, 0.524979, 0, 0,
        0.475021, 0.524979, 0, 0,
        0.30061, 0.332225, 0.367165, 0,
        0.30061, 0.332225, 0.367165, 0
    };
    bool succ = compare_res_ans_1d(res->get_tensor(), ans, "res");
    if (!succ) {
        std::cout << RED << "test_masked_softmax failed" << RESET << std::endl;
    } else {
        std::cout << GREEN << "test_masked_softmax succ" << RESET << std::endl;
    }
    destruct_env();
}

void test_masked_softmax_1() {
    construct_env();
    Tensor *input = allocTensor({2, 2, 4}, "input");
    auto ni = graph::allocNode(input);
    ni->init_weight_for_dbg(10000.0f);
    Tensor *valid_lens = allocTensor({2, 2}, "mask", INT32);
    auto res = ni->masked_softmax(valid_lens);
    insert_boundary_action();
    allocMemAndInitTensors();
    int valid_lens_buffer[4] = {1, 3, 2, 4};
    g_backend_ops->cp_to_device(
        valid_lens,
        reinterpret_cast<char*>(valid_lens_buffer),
        4 * sizeof(int32_t)
    );
    gDoActions();
    float ans[16] = {
        1, 0, 0, 0,
        0.30061, 0.332225, 0.367165, 0,
        0.475021, 0.524979, 0, 0,
        0.213838, 0.236328, 0.261183, 0.288651
    };

    bool succ = compare_res_ans_1d(res->get_tensor(), ans, "res");
    if (!succ) {
        std::cout << RED << "test_masked_softmax_1 failed" << RESET << std::endl;
    } else {
        std::cout << GREEN << "test_masked_softmax_1 succ" << RESET << std::endl;
    }
    destruct_env();
}

void test_masked_softmax_bp() {
    construct_env();
    Tensor *labels = allocTensor({4}, "input", INT32);
    Tensor *input = allocTensor({2, 2, 4}, "input");
    auto ni = graph::allocNode(input);
    ni->require_grad();
    ni->init_weight_for_dbg(10000.0f);
    Tensor *valid_lens = allocTensor({2, 2}, "mask", INT32);
    auto res_softmax = ni->masked_softmax(valid_lens);
    auto res_ce = res_softmax->reshape({-1, 4})->CrossEntropy(labels);
    insert_boundary_action();
    res_ce->backward();
    allocMemAndInitTensors();
    init_labels(labels);
    
    int valid_lens_buffer[4] = {1, 3, 2, 4};
    g_backend_ops->cp_to_device(
        valid_lens,
        reinterpret_cast<char*>(valid_lens_buffer),
        4 * sizeof(int32_t)
    );
    gDoActions();

    float ans[16] = {
        1, 0, 0, 0,
        0.30061, 0.332225, 0.367165, 0,
        0.475021, 0.524979, 0, 0,
        0.213838, 0.236328, 0.261183, 0.288651
    };

    bool succ_softmax = compare_res_ans_1d(res_softmax->get_tensor(), ans, "res_softmax");
    if (!succ_softmax) {
        std::cout << RED << "test_masked_softmax_bp softmax failed" << RESET << std::endl;
    }

    float ni_grad_ans[16] = {
        0, 0, 0, 0,
        0.0242644, -0.0555455, 0.0312811, 0,
        -0.00096927, 0.00096927, 0, 0,
        0.0149098, 0.0168018, 0.0189739, -0.0506855
    };

    bool succ_ni_grad = compare_res_ans_1d(
        ni->get_grad(),
        ni_grad_ans,
        "ni_grad"
    );
    if (!succ_ni_grad) {
        std::cout << RED << "test_masked_softmax_bp ni_grad failed" << RESET << std::endl;
    }

    bool succ = succ_softmax && succ_ni_grad;
    if (succ) {
        std::cout << GREEN << "test_masked_softmax_bp succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_masked_softmax_bp failed" << RESET << std::endl;
    }
    destruct_env();
}

void test_bmm_bp() {

    construct_env();
    Tensor *input = allocTensor({2, 3, 4}, "input");
    auto ni = graph::allocNode(input);
    ni->require_grad();
    ni->init_weight_for_dbg(10000.0f);
    Tensor *w = allocTensor({2, 4, 6}, "w");
    auto nw = graph::allocNode(w);
    nw->require_grad();
    nw->init_weight_for_dbg(10000.0f);

    Tensor *labels = allocTensor({6}, "labels", INT32);
    auto n_labels = graph::allocNode(labels);
    n_labels->init_weight_for_dbg();

    auto bmm_res = ni->bmm(nw);
    auto ce_res = bmm_res->reshape({-1, 6})->CrossEntropy(labels);
    insert_boundary_action();
    ce_res->backward();

    // printAllActions();
    allocMemAndInitTensors();
    gDoActions();
    float loss = 0;
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(&loss),
        ce_res->get_tensor(),
        sizeof(float)
    );
    loss /= 6;

    float loss_ans = 1.6919627f;

    const float eps = 1e-5f;

    bool succ_loss = fabs(loss - loss_ans) < eps;
    if (!succ_loss) {
        std::cerr << RED << "Error: loss = " << loss << ", ans = " << loss_ans << RESET << std::endl;
    }

    float bmm_res_ans[36] = {
        0.84, 0.9, 0.96, 1.02, 1.08, 1.14,
        2.28, 2.5, 2.72, 2.94, 3.16, 3.38,
        3.72, 4.1, 4.48, 4.86, 5.24, 5.62,
        18.12, 18.66, 19.2, 19.74, 20.28, 20.82,
        23.4, 24.1, 24.8, 25.5, 26.2, 26.9,
        28.68, 29.54, 30.4, 31.26, 32.12, 32.98
    };

    bool succ_bmm_res = compare_res_ans_1d(
        bmm_res->get_tensor(),
        bmm_res_ans,
        "bmm_res"
    );

    if (!succ_bmm_res) {
        std::cout << RED << "test_bmm_bp bmm_res failed" << RESET << std::endl;
    }

    auto ni_grad = ni->get_grad();
    auto nw_grad = nw->get_grad();

    float ni_grad_ans[24] = {
        0.0445769, 0.0445769, 0.0445769, 0.0445769,
        0.035388, 0.035388, 0.035388, 0.035388,
        0.025341, 0.025341, 0.025341, 0.025341,
        0.0141321, 0.0141321, 0.0141321, 0.0141321,
        0.00174849, 0.0017485, 0.00174847, 0.00174847,
        -0.011649, -0.011649, -0.011649, -0.011649
    };

    float nw_grad_ans[48] = {
        0.0130027, -0.0489459, -0.109031, 0.0335287, 0.0465271, 0.0649188,
        0.00108722, -0.0599406, -0.118818, 0.0420134, 0.0571685, 0.0784897,
        -0.0108283, -0.0709353, -0.128605, 0.0504981, 0.06781, 0.0920605,
        -0.0227438, -0.0819301, -0.138392, 0.0589828, 0.0784514, 0.105631,
        0.0125765, 0.0245049, 0.0485311, -0.102268, -0.0665401, 0.0831956,
        0.0134513, 0.0261676, 0.0517392, -0.112645, -0.0706674, 0.091954,
        0.0143261, 0.0278303, 0.0549473, -0.123021, -0.0747947, 0.100712,
        0.015201, 0.029493, 0.0581554, -0.133398, -0.0789219, 0.109471
    };

    bool succ_ni_grad = compare_res_ans_1d(
        ni_grad,
        ni_grad_ans,
        "ni_grad"
    );

    if (!succ_ni_grad) {
        std::cout << RED << "test_bmm_bp ni_grad failed" << RESET << std::endl;
    }

    bool succ_nw_grad = compare_res_ans_1d(
        nw_grad,
        nw_grad_ans,
        "nw_grad"
    );

    if (!succ_nw_grad) {
        std::cout << RED << "test_bmm_bp nw_grad failed" << RESET << std::endl;
    }

    bool succ = succ_loss && succ_bmm_res && succ_ni_grad && succ_nw_grad;

    if (succ) {
        std::cout << GREEN << "test_bmm_bp succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_bmm_bp failed" << RESET << std::endl;
    }

    destruct_env();
}

void test_bmm_bp_1() {
    construct_env();
    Tensor *input = allocTensor({2, 3, 4}, "input");
    auto ni = graph::allocNode(input);
    ni->require_grad();
    ni->init_weight_for_dbg(10000.0f);
    Tensor *w = allocTensor({2, 6, 4}, "w");
    auto nw = graph::allocNode(w);
    nw->require_grad();
    nw->init_weight_for_dbg(10000.0f);

    Tensor *labels = allocTensor({6}, "labels", INT32);
    auto n_labels = graph::allocNode(labels);
    n_labels->init_weight_for_dbg();

    auto bmm_res = ni->bmm(nw->transpose(1, 2));
    auto ce_res = bmm_res->reshape({-1, 6})->CrossEntropy(labels);
    insert_boundary_action();
    zero_grad();
    ce_res->backward();

    // printAllActions();
    allocMemAndInitTensors();
    gDoActions();

    float loss = 0;
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(&loss),
        ce_res->get_tensor(),
        sizeof(float)
    );

    // std::cout << "input : " << std::endl << *input << std::endl;
    // std::cout << "w : " << std::endl << *w << std::endl;
    // std::cout << "loss = " << loss / 6 << std::endl;
    // std::cout << "input grad : " << std::endl << *ni->get_grad() << std::endl;
    // std::cout << "w grad : " << std::endl << *nw->get_grad() << std::endl;

    float input_grad_ans[24] = {
        0.211754, 0.211754, 0.211754, 0.211754,
        0.221463, 0.221463, 0.221463, 0.221463,
        0.181381, 0.181381, 0.181381, 0.181381,

        0.124644, 0.124644, 0.124644, 0.124644,
        0.0623503, 0.0623503, 0.0623502, 0.0623502,
        -0.00220847, -0.00220847, -0.00220847, -0.00220847
    };
    float w_grad_ans[48] = {
        0.000533584, -0.0146025, -0.0297386, -0.0448748,
        -0.0652676, -0.0798298, -0.0943921, -0.108954,
        -0.129445, -0.143007, -0.15657, -0.170132,
        0.0117302, 0.0169236, 0.0221169, 0.0273103,
        0.0390515, 0.0496321, 0.0602126, 0.0707932,
        0.143397, 0.170884, 0.198371, 0.225858,
        3.82859e-06, 4.14294e-06, 4.45729e-06, 4.77163e-06,
        3.50633e-05, 3.79026e-05, 4.07418e-05, 4.3581e-05,
        0.00033834, 0.000365007, 0.000391675, 0.000418342,
        -0.196389, -0.212785, -0.229181, -0.245577,
        -0.220686, -0.234183, -0.24768, -0.261177,
        0.416698, 0.446561, 0.476425, 0.506288
    };

    bool succ_input_grad = compare_res_ans_1d(
        ni->get_grad(),
        input_grad_ans,
        "input_grad"
    );

    if (!succ_input_grad) {
        std::cout << RED << "test_bmm_bp_1 input_grad failed" << RESET << std::endl;
    }

    bool succ_w_grad = compare_res_ans_1d(
        nw->get_grad(),
        w_grad_ans,
        "w_grad"
    );

    if (!succ_w_grad) {
        std::cout << RED << "test_bmm_bp_1 w_grad failed" << RESET << std::endl;
    }

    bool succ = succ_input_grad && succ_w_grad;

    if (succ) {
        std::cout << GREEN << "test_bmm_bp_1 succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_bmm_bp_1 failed" << RESET << std::endl;
    }
    destruct_env();
}

void test_div_bp() {
    construct_env();
    Tensor *input = allocTensor({3, 4}, "input");
    auto ni = graph::allocNode(input);
    ni->require_grad();
    ni->init_weight_for_dbg(10000.0f);
    Tensor *w = allocTensor({4, 6}, "w");
    auto nw = graph::allocNode(w);
    nw->require_grad();
    nw->init_weight_for_dbg(10000.0f);

    Tensor *labels = allocTensor({3}, "labels", INT32);
    auto n_labels = graph::allocNode(labels);
    n_labels->init_weight_for_dbg();

    auto res = ni->at(nw)->div(10.0f);
    auto ce_res = res->CrossEntropy(labels);
    ce_res->backward();
    insert_boundary_action();

    allocMemAndInitTensors();
    gDoActions();

    float res_ans[18] = {
        0.084, 0.09, 0.096, 0.102, 0.108, 0.114,
        0.228, 0.25, 0.272, 0.294, 0.316, 0.338,
        0.372, 0.41, 0.448, 0.486, 0.524, 0.562
    };

    float ni_grad_ans[12] = {
        0.00839167, 0.00839167, 0.00839167, 0.00839167,
        0.00521382, 0.00521382, 0.00521382, 0.00521382,
        0.00203578, 0.00203578, 0.00203579, 0.00203579
    };

    float nw_grad_ans[24] = {
        0.00613498, -0.0069954, -0.0201187, 0.00676539, 0.0069904, 0.00722331,
        0.0043785, -0.00871737, -0.0218051, 0.00844891, 0.00871165, 0.00898342,
        0.00262202, -0.0104393, -0.0234915, 0.0101324, 0.0104329, 0.0107435,
        0.00086554, -0.0121613, -0.025178, 0.011816, 0.0121541, 0.0125036
    };

    bool succ_res = compare_res_ans_1d(
        res->get_tensor(),
        res_ans,
        "res"
    );

    if (!succ_res) {
        std::cout << RED << "test_div_bp res failed" << RESET << std::endl;
    }

    bool succ_ni_grad = compare_res_ans_1d(
        ni->get_grad(),
        ni_grad_ans,
        "ni_grad"
    );

    if (!succ_ni_grad) {
        std::cout << RED << "test_div_bp ni_grad failed" << RESET << std::endl;
    }

    bool succ_nw_grad = compare_res_ans_1d(
        nw->get_grad(),
        nw_grad_ans,
        "nw_grad"
    );

    if (!succ_nw_grad) {
        std::cout << RED << "test_div_bp nw_grad failed" << RESET << std::endl;
    }
    

    bool succ = succ_res && succ_ni_grad && succ_nw_grad;

    if (succ) {
        std::cout << GREEN << "test_div_bp succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_div_bp failed" << RESET << std::endl;
    }

    destruct_env();
}

void test_attention_bp_part() {
    construct_env();
    auto d = 2;
    Tensor *querys = allocTensor({2, 1, d}, "querys");
    Tensor *keys = allocTensor({2, 10, d}, "keys");
    // Tensor *values = allocTensor({2, 10, 4}, "values");
    Tensor *valid_lens = allocTensor({2}, "valid_lens", INT32);
    
    Tensor *labels = allocTensor({2}, "labels", INT32);
    auto n_labels = graph::allocNode(labels);
    n_labels->init_weight_for_dbg();
    auto nq = graph::allocNode(querys);
    nq->require_grad();
    nq->init_weight_for_dbg(100.0f);
    auto nk = graph::allocNode(keys);
    nk->require_grad();
    nk->init_weight_for_dbg(100.0f);
    // auto nv = graph::allocNode(values);
    // nv->require_grad();
    // nv->init_weight_for_dbg(10000.0f);

    auto bmm_res = nq->bmm(nk->transpose(1, 2));
    auto softmax_res = bmm_res
        ->div(std::sqrt(static_cast<float>(d)))
        ->masked_softmax(valid_lens);
    auto ce_res = softmax_res->reshape({-1, 10})->CrossEntropy(labels);
    insert_boundary_action();
    zero_grad();
    ce_res->backward();
    // printAllActions();
    allocMemAndInitTensors();
    int32_t valid_lens_buffer[2] = {2, 6};
    g_backend_ops->cp_to_device(
        valid_lens,
        reinterpret_cast<char*>(valid_lens_buffer),
        2 * sizeof(int32_t)
    );
    gDoActions();
    float loss = 0;
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(&loss),
        ce_res->get_tensor(),
        sizeof(float)
    );
    // std::cout << "loss : " << loss/2 << std::endl;
    // std::cout << "labels : " << std::endl << *labels << std::endl;
    // std::cout << "query : " << std::endl << *querys << std::endl;
    // std::cout << "keys : " << std::endl << *keys << std::endl;
    // std::cout << "bmm_res : " << std::endl << *bmm_res->get_tensor() << std::endl;
    // std::cout << "softmax_res : " << std::endl << *softmax_res->get_tensor() << std::endl;
    // // print nq grad nk grad nv grad
    // std::cout << "nq grad : " << std::endl << *nq->get_grad() << std::endl;
    // std::cout << "nk grad : " << std::endl << *nk->get_grad() << std::endl;
    // std::cout << "bmm grad : " << std::endl << *bmm_res->get_grad() << std::endl;
    // std::cout << "softmax_res grad : " << std::endl << *softmax_res->get_grad() << std::endl;

    float softmax_res_ans[20] = {
        0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0,
        0.166664, 0.166665, 0.166666, 0.166667, 0.166668, 0.16667, 0, 0, 0, 0
    };

    float nq_grad_ans[4] = {
        0.000176777, 0.000176777,
        0.000176777, 0.000176777
    };

    float nk_grad_ans[40] = {
        0, -8.83884e-05,
        0, 8.83884e-05,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        1.96413e-05, 2.94619e-05,
        -9.82085e-05, -0.000147313,
        1.96416e-05, 2.94624e-05,
        1.96417e-05, 2.94626e-05,
        1.96419e-05, 2.94628e-05,
        1.9642e-05, 2.9463e-05,
        0, 0,
        0, 0,
        0, 0,
        0, 0
    };

    bool succ_softmax_res = compare_res_ans_1d(
        softmax_res->get_tensor(),
        softmax_res_ans,
        "softmax_res"
    );

    if (!succ_softmax_res) {
        std::cout << RED << "test_attention_bp_part softmax_res failed" << RESET << std::endl;
    }

    bool succ_nq_grad = compare_res_ans_1d(
        nq->get_grad(),
        nq_grad_ans,
        "nq_grad"
    );

    if (!succ_nq_grad) {
        std::cout << RED << "test_attention_bp_part nq_grad failed" << RESET << std::endl;
    }

    bool succ_nk_grad = compare_res_ans_1d(
        nk->get_grad(),
        nk_grad_ans,
        "nk_grad"
    );

    if (!succ_nk_grad) {
        std::cout << RED << "test_attention_bp_part nk_grad failed" << RESET << std::endl;
    }

    bool succ = succ_softmax_res && succ_nq_grad && succ_nk_grad;

    if (succ) {
        std::cout << GREEN << "test_attention_bp_part succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_attention_bp_part failed" << RESET << std::endl;
    }
    destruct_env();
}

void test_attention_bp() {
    construct_env();
    DotProductAttention attention;
    Tensor *querys = allocTensor({2, 1, 2}, "querys");
    Tensor *keys = allocTensor({2, 10, 2}, "keys");
    Tensor *values = allocTensor({2, 10, 4}, "values");
    Tensor *valid_lens = allocTensor({2}, "valid_lens", INT32);

    Tensor *labels = allocTensor({2}, "labels", INT32);
    auto n_labels = graph::allocNode(labels);
    n_labels->init_weight_for_dbg();
    auto nq = graph::allocNode(querys);
    nq->require_grad();
    // nq->init_weight_for_dbg(1000000.0f);
    auto nk = graph::allocNode(keys);
    nk->require_grad();
    // nk->init_weight_for_dbg(1000000.0f);
    auto nv = graph::allocNode(values);
    nv->require_grad();
    nv->init_weight_for_dbg(10000.0f);
    int32_t valid_lens_buffer[2] = {2, 6};
    auto softmax_res = attention.forward(nq, nk, nv, valid_lens)->softmax();
    auto ce_res = softmax_res->reshape({-1, 4})->CrossEntropy(labels);
    zero_grad();
    insert_boundary_action();
    ce_res->backward();
    // printAllActions();
    allocMemAndInitTensors();
    querys->fill(10.6f);
    keys->fill(55.5f);
    g_backend_ops->cp_to_device(
        valid_lens,
        reinterpret_cast<char*>(valid_lens_buffer),
        2 * sizeof(int32_t)
    );
    gDoActions();

    float loss = 0;
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(&loss),
        ce_res->get_tensor(),
        sizeof(float)
    );

    float softmax_res_ans[8] = {
        0.213838, 0.236328, 0.261183, 0.288651,
        0.213838, 0.236328, 0.261183, 0.288651
    };

    float nq_grad_ans[4] = {
        0, 0,
        -7.42405e-09, -7.42405e-09
    };

    float nk_grad_ans[40] = {
        -6.98057e-09, -6.98057e-09,
        6.98057e-09, 6.98057e-09,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        2.73769e-08, 2.73769e-08,
        -2.80314e-08, -2.80314e-08,
        2.73769e-08, 2.73769e-08,
        -4.67916e-08, -4.67916e-08,
        -9.27107e-09, -9.27107e-09,
        2.79223e-08, 2.79223e-08,
        0, 0,
        0, 0,
        0, 0,
        0, 0
    };

    float nv_grad_ans[80] = {
        -0.0425492, 0.0123817, 0.014089, 0.0160786,
        -0.0425492, 0.0123817, 0.014089, 0.0160786,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0.00403754, -0.0151238, 0.00518581, 0.00590049,
        0.00403754, -0.0151238, 0.00518581, 0.00590049,
        0.00403754, -0.0151238, 0.00518581, 0.00590049,
        0.00403754, -0.0151238, 0.00518581, 0.00590049,
        0.00403754, -0.0151238, 0.00518581, 0.00590049,
        0.00403754, -0.0151238, 0.00518581, 0.00590049,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    bool succ_softmax_res = compare_res_ans_1d(
        softmax_res->get_tensor(),
        softmax_res_ans,
        "softmax_res"
    );

    if (!succ_softmax_res) {
        std::cout << RED << "test_attention_bp softmax_res failed" << RESET << std::endl;
    }

    bool succ_nq_grad = compare_res_ans_1d(
        nq->get_grad(),
        nq_grad_ans,
        "nq_grad"
    );

    if (!succ_nq_grad) {
        std::cout << RED << "test_attention_bp nq_grad failed" << RESET << std::endl;
    }

    bool succ_nk_grad = compare_res_ans_1d(
        nk->get_grad(),
        nk_grad_ans,
        "nk_grad"
    );

    if (!succ_nk_grad) {
        std::cout << RED << "test_attention_bp nk_grad failed" << RESET << std::endl;
    }

    bool nv_grad = compare_res_ans_1d(
        nv->get_grad(),
        nv_grad_ans,
        "nv_grad"
    );

    if (!nv_grad) {
        std::cout << RED << "test_attention_bp nv_grad failed" << RESET << std::endl;
    }

    bool succ = succ_softmax_res && succ_nq_grad && succ_nk_grad && nv_grad;
    if (succ) {
        std::cout << GREEN << "test_attention_bp succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_attention_bp failed" << RESET << std::endl;
    }
    destruct_env();
}

void test_dropout() {
    construct_env();
    Dropout dropout(0.5f);
    Tensor *input = allocTensor({22, 33, 55}, "input");
    // Tensor *input = allocTensor({2, 3, 5}, "input");
    auto input_shape = input->get_shape();
    auto ni = graph::allocNode(input);
    ni->require_grad();
    auto res = dropout.forward(ni)->reshape(input_shape);
    res->backward();
    insert_boundary_action();
    allocMemAndInitTensors();
    input->fill(1.0f);
    res->get_grad()->fill(1.0f);
    // printAllActions();
    gDoActions();
    float *res_buffer = static_cast<float*>(::malloc(res->get_tensor()->size()));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_buffer),
        res->get_tensor(),
        res->get_tensor()->size()
    );
    float sum = 0;
    auto length = res->get_tensor()->length();
    for (int i = 0; i < length; ++ i) {
        sum += res_buffer[i];
    }
    float percent = sum / length;
    bool succ_res = percent > 0.4f && percent < 0.6f;
    if (!succ_res) {
        std::cout << RED << "test_dropout res failed" << RESET << std::endl;
    }

    float *ni_grad_buffer = static_cast<float*>(::malloc(ni->get_grad()->size()));
    auto ni_grad_length = ni->get_grad()->length();
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(ni_grad_buffer),
        ni->get_grad(),
        ni->get_grad()->size()
    );
    sum = 0;
    for (int i = 0; i < ni_grad_length; ++ i) {
        sum += ni_grad_buffer[i];
    }
    percent = sum / ni_grad_length;
    bool succ_grad = percent > 0.4f && percent < 0.6f;
    if (!succ_grad) {
        std::cout << RED << "test_dropout ni_grad failed" << RESET << std::endl;
    }

    bool succ = succ_res && succ_grad;
    if (succ) {
        std::cout << GREEN << "test_dropout succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_dropout failed" << RESET << std::endl;
    }
    ::free(res_buffer);
    ::free(ni_grad_buffer);
    destruct_env();
}

void test_permute() {
    construct_env();
    Tensor *input = allocTensor({2, 3, 4, 5}, "input");
    Tensor *w = allocTensor({5, 4}, "w");
    auto ni = graph::allocNode(input);
    auto nw = graph::allocNode(w);
    nw->init_weight_for_dbg(10000.0f);
    ni->require_grad();
    auto p_res = ni->permute({2, 0, 1, 3});
    auto r_res = p_res->reshape({-1, 5});
    auto w_res = r_res->at(nw);
    insert_boundary_action();
    w_res->backward();
    // printAllActions();
    allocMemAndInitTensors();
    float grad_buffer[96] = {0};
    assert(w_res->get_grad()->length() == 96);
    for (int i = 0; i < w_res->get_grad()->length(); ++ i) {
       grad_buffer[i] = 1.0f;
    }

    g_backend_ops->cp_to_device(
        w_res->get_grad(),
        reinterpret_cast<char*>(grad_buffer),
        96 * sizeof(float)
    );

    float input_buffer[120] = {
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5,
        1.6, 1.7, 1.8, 1.9, 2.0,
        2.1, 2.2, 2.3, 2.4, 2.5,
        2.6, 2.7, 2.8, 2.9, 3.0,
        3.1, 3.2, 3.3, 3.4, 3.5,
        3.6, 3.7, 3.8, 3.9, 4.0,
        4.1, 4.2, 4.3, 4.4, 4.5,
        4.6, 4.7, 4.8, 4.9, 5.0,
        5.0, 5.1, 5.2, 5.3, 5.4,
        5.5, 5.6, 5.7, 5.8, 5.9,
        6.0, 6.1, 6.2, 6.3, 6.4,
        6.5, 6.6, 6.7, 6.8, 6.9,
        7.0, 7.1, 7.2, 7.3, 7.4,
        7.5, 7.6, 7.7, 7.8, 7.9,
        8.0, 8.1, 8.2, 8.3, 8.4,
        8.5, 8.6, 8.7, 8.8, 8.9,
        9.0, 9.1, 9.2, 9.3, 9.4,
        9.5, 9.6, 9.7, 9.8, 9.9,
        10.0, 10.1, 10.2, 10.3, 10.4,
        10.5, 10.6, 10.7, 10.8, 10.9,
        11.0, 11.1, 11.2, 11.3, 11.4,
        11.5, 11.6, 11.7, 11.8, 11.9
    };

    float input_p_buffer[120] = {
        0.1000,  0.2000,  0.3000,  0.4000,  0.5000,
        2.1000,  2.2000,  2.3000,  2.4000,  2.5000,
        4.1000,  4.2000,  4.3000,  4.4000,  4.5000,
        6.0000,  6.1000,  6.2000,  6.3000,  6.4000,
        8.0000,  8.1000,  8.2000,  8.3000,  8.4000,
        10.0000, 10.1000, 10.2000, 10.3000, 10.4000,
        0.6000,  0.7000,  0.8000,  0.9000,  1.0000,
        2.6000,  2.7000,  2.8000,  2.9000,  3.0000,
        4.6000,  4.7000,  4.8000,  4.9000,  5.0000,
        6.5000,  6.6000,  6.7000,  6.8000,  6.9000,
        8.5000,  8.6000,  8.7000,  8.8000,  8.9000,
        10.5000, 10.6000, 10.7000, 10.8000, 10.9000,
        1.1000,  1.2000,  1.3000,  1.4000,  1.5000,
        3.1000,  3.2000,  3.3000,  3.4000,  3.5000,
        5.0000,  5.1000,  5.2000,  5.3000,  5.4000,
        7.0000,  7.1000,  7.2000,  7.3000,  7.4000,
        9.0000,  9.1000,  9.2000,  9.3000,  9.4000,
        11.0000, 11.1000, 11.2000, 11.3000, 11.4000,
        1.6000,  1.7000,  1.8000,  1.9000,  2.0000,
        3.6000,  3.7000,  3.8000,  3.9000,  4.0000,
        5.5000,  5.6000,  5.7000,  5.8000,  5.9000,
        7.5000,  7.6000,  7.7000,  7.8000,  7.9000,
        9.5000,  9.6000,  9.7000,  9.8000,  9.9000,
        11.5000, 11.6000, 11.7000, 11.8000, 11.9000
    };
    g_backend_ops->cp_to_device(
        input,
        reinterpret_cast<char*>(input_buffer),
        input->size()
    );

    gDoActions();
    bool succ_permute = compare_res_ans_1d(
        r_res->get_tensor(),
        input_p_buffer,
        "res"
    );

    if (!succ_permute) {
        std::cout << RED << "test_permute res failed" << RESET << std::endl;
    }

    float w_res_ans[96] = {
        1.6000,  1.7500,  1.9000,  2.0500,
        9.6000, 10.7500, 11.9000, 13.0500,
        17.6000, 19.7500, 21.9000, 24.0500,
        25.2000, 28.3000, 31.4000, 34.5000,
        33.2000, 37.3000, 41.4000, 45.5000,
        41.2000, 46.3000, 51.4000, 56.5000,
        3.6000,  4.0000,  4.4000,  4.8000,
        11.6000, 13.0000, 14.4000, 15.8000,
        19.6000, 22.0000, 24.4000, 26.8000,
        27.2000, 30.5500, 33.9000, 37.2500,
        35.2000, 39.5500, 43.9000, 48.2500,
        43.2000, 48.5500, 53.9000, 59.2500,
        5.6000,  6.2500,  6.9000,  7.5500,
        13.6000, 15.2500, 16.9000, 18.5500,
        21.2000, 23.8000, 26.4000, 29.0000,
        29.2000, 32.8000, 36.4000, 40.0000,
        37.2000, 41.8000, 46.4000, 51.0000,
        45.2000, 50.8000, 56.4000, 62.0000,
        7.6000,  8.5000,  9.4000, 10.3000,
        15.6000, 17.5000, 19.4000, 21.3000,
        23.2000, 26.0500, 28.9000, 31.7500,
        31.2000, 35.0500, 38.9000, 42.7500,
        39.2000, 44.0500, 48.9000, 53.7500,
        47.2000, 53.0500, 58.9000, 64.7500
    };

    bool succ_w = compare_res_ans_1d(
        w_res->get_tensor(),
        w_res_ans,
        "w_res"
    );

    if (!succ_w) {
        std::cout << RED << "test_permute w_res failed" << RESET << std::endl;
    }

    bool succ = succ_permute && succ_w;
    if (!succ) {
        std::cout << RED << "test_permute res failed" << RESET << std::endl;
    } else {
        std::cout << GREEN << "test_permute succ" << RESET << std::endl;
    }
    destruct_env();
}

void test_lazy_linear() {

    construct_env();
    Tensor *input = allocTensor({20, 30, 10}, "input");
    Tensor *input1 = allocTensor({20 * 30, 10}, "input");
    auto ni = graph::allocNode(input);
    auto ni1 = graph::allocNode(input1);

    LazyLinear l(20, "", 0, 0, ACTIVATION::NONE, false, true);

    auto res = l.forward(ni);
    auto res1 = l.forward(ni1);

    auto res_buffer = static_cast<float*>(::malloc(res->get_tensor()->size()));
    auto res1_buffer = static_cast<float*>(::malloc(res1->get_tensor()->size()));

    insert_boundary_action();
    allocMemAndInitTensors();
    gDoForwardActions();

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_buffer),
        res->get_tensor(),
        res->get_tensor()->size()
    );

    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res1_buffer),
        res1->get_tensor(),
        res1->get_tensor()->size()
    );

    bool succ = compare_ans1_ans2(
        res_buffer,
        res1_buffer,
        res->get_tensor()->length()
    );

    if (!succ) {
        std::cout << RED << "test_lazy_linear res failed" << RESET << std::endl;
    } else {
        std::cout << GREEN << "test_lazy_linear succ" << RESET << std::endl;
    }
    destruct_env();

    ::free(res_buffer);
    ::free(res1_buffer);
}

void test_mha() {
    construct_env();

    Tensor *queries = allocTensor({2, 1, 2}, "queries");

    float queries_buffer[2 * 1 * 2] = {
        0.1, 0.1,
        0.2, 0.2
    };

    Tensor *keys = allocTensor({2, 5, 2}, "keys");

    float keys_buffer[2 * 5 * 2] = {
        1.1, 1.1,
        1.2, 1.2,
        1.3, 1.3,
        1.4, 1.4,
        1.5, 1.5,
        2.1, 2.1,
        2.2, 2.2,
        2.3, 2.3,
        2.4, 2.4,
        2.5, 2.5
    };

    Tensor *values = allocTensor({2, 5, 4}, "values");

    float values_buffer[2 * 5 * 4] = {
        3.1, 3.1, 3.1, 3.1,
        3.2, 3.2, 3.2, 3.2,
        3.3, 3.3, 3.3, 3.3,
        3.4, 3.4, 3.4, 3.4,
        3.5, 3.5, 3.5, 3.5,
        4.1, 4.1, 4.1, 4.1,
        4.2, 4.2, 4.2, 4.2,
        4.3, 4.3, 4.3, 4.3,
        4.4, 4.4, 4.4, 4.4,
        4.5, 4.5, 4.5, 4.5
    };

    auto nq = graph::allocNode(queries);
    auto nk = graph::allocNode(keys);
    auto nv = graph::allocNode(values);

    nq->require_grad();
    nk->require_grad();
    nv->require_grad();

    Tensor *labels = allocTensor({2}, "labels", INT32);
    int32_t labels_buffer[2] = {0, 0};
    Tensor *valid_lens = allocTensor({2}, "valid_lens", INT32);
    int32_t valid_lens_buffer[2] = {2, 4};

    MHA mha(10, 2, 0.0f, false, true);
    auto res = mha.forward(nq, nk, nv, valid_lens);
    auto res_shape = res->get_tensor()->get_shape();
    auto res_dim = res->get_tensor()->get_dim();
    
    auto ce_res = res->reshape({-1, res_shape[res_dim-1]})->CrossEntropy(labels);
    insert_boundary_action();
    ce_res->backward();
    // printAllActions();
    allocMemAndInitTensors();

    std::vector<Parameter *> params = mha.get_parameters();
    auto w_q_w_linear = params[0]->get_w();
    auto w_k_w_linear = params[1]->get_w();
    auto w_v_w_linear = params[2]->get_w();
    auto w_o_w_linear = params[3]->get_w();

    float *w_q_w_linear_buffer = static_cast<float*>(::malloc(w_q_w_linear->size()));
    float *w_k_w_linear_buffer = static_cast<float*>(::malloc(w_k_w_linear->size()));
    float *w_v_w_linear_buffer = static_cast<float*>(::malloc(w_v_w_linear->size()));
    float *w_o_w_linear_buffer = static_cast<float*>(::malloc(w_o_w_linear->size()));

    for (int i = 0; i < w_q_w_linear->length(); ++ i) {
        w_q_w_linear_buffer[i] = 1.0f;
    }
    for (int i = 0; i < w_k_w_linear->length(); ++ i) {
        w_k_w_linear_buffer[i] = 1.0f;
    }
    for (int i = 0; i < w_v_w_linear->length(); ++ i) {
        w_v_w_linear_buffer[i] = 1.0f;
    }
    for (int i = 0; i < w_o_w_linear->length(); ++ i) {
        w_o_w_linear_buffer[i] = 1.0f;
    }

    w_q_w_linear_buffer[0] = 0.1f;
    w_k_w_linear_buffer[0] = 0.1f;
    w_v_w_linear_buffer[0] = 0.1f;
    w_o_w_linear_buffer[0] = 0.1f;

    g_backend_ops->cp_to_device(
        queries,
        reinterpret_cast<char*>(queries_buffer),
        queries->size()
    );
    g_backend_ops->cp_to_device(
        keys,
        reinterpret_cast<char*>(keys_buffer),
        keys->size()
    );
    g_backend_ops->cp_to_device(
        values,
        reinterpret_cast<char*>(values_buffer),
        values->size()
    );
    g_backend_ops->cp_to_device(
        labels,
        reinterpret_cast<char*>(labels_buffer),
        labels->size()
    );
    g_backend_ops->cp_to_device(
        valid_lens,
        reinterpret_cast<char*>(valid_lens_buffer),
        valid_lens->size()
    );
    g_backend_ops->cp_to_device(
        w_q_w_linear,
        reinterpret_cast<char*>(w_q_w_linear_buffer),
        w_q_w_linear->size()
    );
    g_backend_ops->cp_to_device(
        w_k_w_linear,
        reinterpret_cast<char*>(w_k_w_linear_buffer),
        w_k_w_linear->size()
    );
    g_backend_ops->cp_to_device(
        w_v_w_linear,
        reinterpret_cast<char*>(w_v_w_linear_buffer),
        w_v_w_linear->size()
    );
    g_backend_ops->cp_to_device(
        w_o_w_linear,
        reinterpret_cast<char*>(w_o_w_linear_buffer),
        w_o_w_linear->size()
    );
    ::free(w_q_w_linear_buffer);
    ::free(w_k_w_linear_buffer);
    ::free(w_v_w_linear_buffer);
    ::free(w_o_w_linear_buffer);

    disableOnceAction();
    gDoActions();

    float res_ans[20] = {
        114.45257, 123.24643, 123.24643, 123.24643, 123.24643, 123.24643, 123.24643, 123.24643, 123.24643, 123.24643,
        155.07243, 166.98326, 166.98326, 166.98326, 166.98326, 166.98326, 166.98326, 166.98326, 166.98326, 166.98326
    };

    assert(res->get_tensor()->length() == 20);
    bool succ_res = compare_res_ans_1d(
        res->get_tensor(),
        res_ans,
        "res"
    );
    if (!succ_res) {
        std::cout << RED << "test_mha res failed" << RESET << std::endl;
    }

    float loss = 0;
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(&loss),
        ce_res->get_tensor(),
        sizeof(float)
    );
    loss /= res_shape[0];
    bool succ_loss = fabs(loss - 12.549578) < 1e-5;
    if (!succ_loss) {
        std::cout << RED << "test_mha loss failed" << RESET << std::endl;
    }

    float nq_grad_ans[4] = {
        0.012630426, 0.01417224,
        0.061989211, 0.069556311
    };

    assert(nq->get_grad()->length() == 4);
    bool succ_nq_grad = compare_res_ans_1d(
        nq->get_grad(),
        nq_grad_ans,
        "nq_grad"
    );

    if (!succ_nq_grad) {
        std::cout << RED << "test_mha nq_grad failed" << RESET << std::endl;
    }

    float nk_grad_ans[20] = {
        -0.012629911, -0.014171663,
        0.012629954, 0.014171709,
        0, 0,
        0, 0,
        0, 0,
        -0.033463478, -0.037548412,
        -0.015951805, -0.017899064,
        0.0083152084, 0.0093302568,
        0.041100066, 0.046117205,
        0,0
    };

    assert(nk->get_grad()->length() == 20);
    bool succ_nk_grad = compare_res_ans_1d(
        nk->get_grad(),
        nk_grad_ans,
        "nk_grad"
    );

    if (!succ_nk_grad) {
        std::cout << RED << "test_mha nk_grad failed" << RESET << std::endl;
    }

    float nv_grad_ans[40] = {
        0.021634489, 0.2163423, 0.2163423, 0.2163423,
        0.023365362, 0.2336507, 0.2336507, 0.2336507,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0.0087997541, 0.087996542, 0.087996542, 0.087996542,
        0.010264132, 0.10264011, 0.10264011, 0.10264011,
        0.011972192, 0.11972046, 0.11972046, 0.11972046,
        0.013964491, 0.13964315, 0.13964315, 0.13964315,
        0, 0, 0, 0
    };

    bool succ_nv_grad = compare_res_ans_1d(
        nv->get_grad(),
        nv_grad_ans,
        "nv_grad"
    );

    if (!succ_nv_grad) {
        std::cout << RED << "test_mha nv_grad failed" << RESET << std::endl;
    }

    bool succ = succ_res && succ_loss && succ_nq_grad && succ_nk_grad && succ_nv_grad;

    if (succ) {
        std::cout << GREEN << "test_mha succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_mha failed" << RESET << std::endl;
    }
    destruct_env();   
}

void test_embedding() {
    construct_env();
    Tensor *indices = allocTensor({3}, "indices", INT32);
    Embedding emb(10, 5, true);
    auto res = emb.forward(indices);
    auto res_grad = res->get_grad();
    insert_boundary_action();
    res->backward();
    // printAllActions();
    allocMemAndInitTensors();
    int32_t indices_buffer[3] = {5, 2, 0};
    g_backend_ops->cp_to_device(
        indices,
        reinterpret_cast<char*>(indices_buffer),
        indices->size()
    );
    auto res_grad_buffer = static_cast<float*>(::malloc(res_grad->size()));
    for (int i = 0; i < res_grad->length(); ++ i) {
        res_grad_buffer[i] = 1.0f * i;
    }
    g_backend_ops->cp_to_device(
        res_grad,
        reinterpret_cast<char*>(res_grad_buffer),
        res_grad->size()
    );
    ::free(res_grad_buffer);
    gDoActions();

    float res_ans[15] = {
        2.5, 2.6, 2.7, 2.8, 2.9,
        1, 1.1, 1.2, 1.3, 1.4,
        0, 0.1, 0.2, 0.3, 0.4
    };
    bool succ_res = compare_res_ans_1d(
        res->get_tensor(),
        res_ans,
        "res"
    );

    if (!succ_res) {
        std::cout << RED << "test_embedding res failed" << RESET << std::endl;
    }

    float grad_ans[50] = {
        10, 11, 12, 13, 14,
        0, 0, 0, 0, 0,
        5, 6, 7, 8, 9,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 1, 2, 3, 4,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    };

    bool succ_grad = compare_res_ans_1d(
        emb.get_grad(),
        grad_ans,
        "grad"
    );

    if (!succ_grad) {
        std::cout << RED << "test_embedding grad failed" << RESET << std::endl;
    }

    bool succ = succ_res && succ_grad;

    if (succ) {
        std::cout << GREEN << "test_embedding succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_embedding failed" << RESET << std::endl;
    }
    destruct_env();
}

void test_pe() {
    construct_env();
    PosEncoding pe(1000, 20);
    Tensor *input = allocTensor({2, 20}, "input");
    auto ni = graph::allocNode(input);
    ni->require_grad();
    ni->init_weight_fill(1.0f);
    auto res = pe.forward(ni);
    insert_boundary_action();
    res->backward();
    // printAllActions();
    allocMemAndInitTensors();
    gDoActions();
    float res_ans[40] = {
        1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
        1.841471, 1.5403023, 1.3876742, 1.9217964, 1.1578267, 1.9874668, 1.0630538, 1.9980102, 1.0251162, 1.9996846, 1.0099999, 1.9999499, 1.0039811, 1.9999921, 1.0015849, 1.9999988, 1.000631, 1.9999998, 1.0002512, 2
    };
    bool succ = compare_res_ans_1d(
        res->get_tensor(),
        res_ans,
        "res"
    );
    if (succ) {
        std::cout << GREEN << "test_pe succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_pe failed" << RESET << std::endl;
    }
    destruct_env();
}

void test_cpu() {
    test_at();
    test_add();
    test_add_eq();
    test_expand_add();
    test_mul();
    test_mul_1();
    test_sum();
    test_cross_entropy();
    test_cross_entropy_backward();
    test_bp();
    test_adam();
    test_mlp();
    // test_print_tensor();
    test_contiguous();
    test_reshape();
    test_reshape_1();
    test_reshape_bp();
    test_reshape_bp_1();
    test_repeat_interleave();
    test_mask();
    test_mask_1();
    test_softmax();
    test_masked_softmax();
    test_masked_softmax_1();
    test_masked_softmax_bp();
    test_bmm();
    test_bmm_1();
    test_bmm_2();
    test_bmm_bp();
    test_div_bp();
    test_bmm_bp_1();
    test_attention_bp();
    test_attention_bp_part();
    test_dropout();
    test_permute();
    test_lazy_linear();
    test_mha();
    test_embedding();
    test_pe();
}

Tensor *test_add_with_cpu_base(int m, int n) {
    Tensor *input = allocTensor({m, n}, "input");
    Tensor *w = allocTensor({m, n}, "w");
    Tensor *res_wi_tensor = allocTensor({m, n}, "res_wi");
    gCreateAction(
        new AddAction(input, w, res_wi_tensor)
    );
    insert_boundary_action();
    allocMemAndInitTensors();
    input->fill(0.1f);
    std::vector<int> w_strides = w->get_strides();
    float *w_tmp_buffer = static_cast<float*>(::malloc(w->size()));
    auto shape = w->get_shape();
    for (int i = 0; i < shape[0]; ++ i) {
        for (int j = 0; j < shape[1]; ++ j) {
            float *loc_w_tmp = w_tmp_buffer + i * w_strides[0] + j * w_strides[1];
            float v = i * shape[1] + j;
            *loc_w_tmp = v;
        }
    }
    g_backend_ops->cp_to_device(
        w,
        reinterpret_cast<char*>(w_tmp_buffer),
        w->size()
    );
    ::free(w_tmp_buffer);
    gDoActions();
    return res_wi_tensor;
}

Tensor *test_at_with_cpu_base(int m, int n, int p) {
    Tensor *input = allocTensor({m, n}, "input");
    Tensor *w = allocTensor({n, p}, "w");
    Tensor *res_wi_tensor = allocTensor({m, p}, "res_wi");
    gCreateAction(
        new AtAction(input, w, res_wi_tensor)
    );
    insert_boundary_action();
    allocMemAndInitTensors();
    input->fill(0.1f);
    std::vector<int> w_strides = w->get_strides();
    float *w_tmp_buffer = static_cast<float*>(::malloc(w->size()));
    auto shape = w->get_shape();
    for (int i = 0; i < shape[0]; ++ i) {
        for (int j = 0; j < shape[1]; ++ j) {
            float *loc_w_tmp = w_tmp_buffer + i * w_strides[0] + j * w_strides[1];
            float v = i * shape[1] + j;
            *loc_w_tmp = v;
        }
    }
    g_backend_ops->cp_to_device(
        w,
        reinterpret_cast<char*>(w_tmp_buffer),
        w->size()
    );
    ::free(w_tmp_buffer);
    gDoActions();
    return res_wi_tensor;
}

void test_gpu_add_with_cpu() {
    use_gpu(false);
    construct_env();
    int m = 330;
    int n = 620;
    // int m = 10;
    // int n = 2;
    Tensor *cpu_res = test_add_with_cpu_base(m, n);
    auto cpu_res_size = cpu_res->size();
    auto cpu_res_length = cpu_res->length();
    float *cpu_res_buffer = static_cast<float*>(::malloc(cpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(cpu_res_buffer),
        cpu_res,
        cpu_res_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    Tensor *gpu_res = test_add_with_cpu_base(m, n);
    auto gpu_res_size = gpu_res->size();
    auto gpu_res_length = gpu_res->length();
    float *gpu_res_buffer = static_cast<float*>(::malloc(gpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(gpu_res_buffer),
        gpu_res,
        gpu_res_size
    );
    destruct_env();
    assert(cpu_res_size == gpu_res_size);
    assert(cpu_res_length == gpu_res_length);
    const float eps = 1e-5f;
    //compare cpu and gpu result
    bool succ = true;
    for (int i = 0; i < cpu_res_length; ++ i) {
        if (fabs(cpu_res_buffer[i] - gpu_res_buffer[i]) > eps) {
            std::cerr << RED << "Error: cpu_res[" << i << "] = " << cpu_res_buffer[i]
                      << ", gpu_res[" << i << "] = " << gpu_res_buffer[i] << RESET << std::endl;
            succ = false;
            break;
        }
    }
    if (succ) {
        std::cout << GREEN << "test_add_with_cpu succ" << RESET << std::endl;
    }
    ::free(gpu_res_buffer);
    ::free(cpu_res_buffer);
}

void test_gpu_at_with_cpu() {
    use_gpu(false);
    construct_env();
    int m = 103;
    int n = 80;
    int p = 102;
    Tensor *cpu_res = test_at_with_cpu_base(m, n, p);
    auto cpu_res_size = cpu_res->size();
    auto cpu_res_length = cpu_res->length();
    float *cpu_res_buffer = static_cast<float*>(::malloc(cpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(cpu_res_buffer),
        cpu_res,
        cpu_res_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    Tensor *gpu_res = test_at_with_cpu_base(m, n, p);
    auto gpu_res_size = gpu_res->size();
    auto gpu_res_length = gpu_res->length();
    float *gpu_res_buffer = static_cast<float*>(::malloc(gpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(gpu_res_buffer),
        gpu_res,
        gpu_res_size
    );
    destruct_env();
    assert(cpu_res_size == gpu_res_size);
    assert(cpu_res_length == gpu_res_length);
    const float eps = 1e-3f;
    //compare cpu and gpu result
    float sum = 0.0f;
    for (int i = 0; i < cpu_res_length; ++ i) {
        float diff = cpu_res_buffer[i] - gpu_res_buffer[i];
        sum += std::pow(diff, 2);
    }
    float rsme = std::sqrt(sum / cpu_res_length);
    bool succ = rsme < eps;
    if (succ) {
        std::cout << GREEN << "test_at_with_cpu succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_at_with_cpu failed, rsme = " << rsme << RESET << std::endl;
    }
    ::free(gpu_res_buffer);
    ::free(cpu_res_buffer);
}

Tensor *test_add_eq_1d_with_cpu_base(int m) {
    Tensor *input = allocTensor({m}, "input");
    Tensor *w = allocTensor({m}, "w");
    gCreateAction(
        new AddEqAction(input, w)
    );
    insert_boundary_action();
    allocMemAndInitTensors();
    input->fill(0.1f);
    std::vector<int> w_strides = w->get_strides();
    float *w_tmp_buffer = static_cast<float*>(::malloc(w->size()));
    auto shape = w->get_shape();
    for (int i = 0; i < shape[0]; ++ i) {
        float *loc_w_tmp = w_tmp_buffer + i * w_strides[0];
        float v = i;
        *loc_w_tmp = v;
    }
    g_backend_ops->cp_to_device(
        w,
        reinterpret_cast<char*>(w_tmp_buffer),
        w->size()
    );
    ::free(w_tmp_buffer);
    gDoActions();
    return input;
}

Tensor *test_add_eq_2d_with_cpu_base(int m, int n) {
    Tensor *input = allocTensor({m, n}, "input");
    Tensor *w = allocTensor({m, n}, "w");
    gCreateAction(
        new AddEqAction(input, w)
    );
    insert_boundary_action();
    allocMemAndInitTensors();
    input->fill(0.1f);
    std::vector<int> w_strides = w->get_strides();
    float *w_tmp_buffer = static_cast<float*>(::malloc(w->size()));
    auto shape = w->get_shape();
    for (int i = 0; i < shape[0]; ++ i) {
        for (int j = 0; j < shape[1]; ++ j) {
            float *loc_w_tmp = w_tmp_buffer + i * w_strides[0] + j * w_strides[1];
            float v = i * shape[1] + j;
            *loc_w_tmp = v;
        }
    }
    g_backend_ops->cp_to_device(
        w,
        reinterpret_cast<char*>(w_tmp_buffer),
        w->size()
    );
    ::free(w_tmp_buffer);
    gDoActions();
    return input;
}

void test_gpu_add_eq_1d_with_cpu() {
    use_gpu(false);
    construct_env();
    int m = 103;
    Tensor *cpu_res = test_add_eq_1d_with_cpu_base(m);
    auto cpu_res_size = cpu_res->size();
    auto cpu_res_length = cpu_res->length();
    float *cpu_res_buffer = static_cast<float*>(::malloc(cpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(cpu_res_buffer),
        cpu_res,
        cpu_res_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    Tensor *gpu_res = test_add_eq_1d_with_cpu_base(m);
    auto gpu_res_size = gpu_res->size();
    auto gpu_res_length = gpu_res->length();
    float *gpu_res_buffer = static_cast<float*>(::malloc(gpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(gpu_res_buffer),
        gpu_res,
        gpu_res_size
    );
    destruct_env();
    assert(cpu_res_size == gpu_res_size);
    assert(cpu_res_length == gpu_res_length);
    const float eps = 1e-5f;
    //compare cpu and gpu result
    bool succ = true;
    for (int i = 0; i < cpu_res_length; ++ i) {
        if (fabs(cpu_res_buffer[i] - gpu_res_buffer[i]) > eps) {
            std::cerr << RED << "Error: cpu_res[" << i << "] = " << cpu_res_buffer[i]
                      << ", gpu_res[" << i << "] = " << gpu_res_buffer[i] << RESET << std::endl;
            succ = false;
            break;
        }
    }
    if (succ) {
        std::cout << GREEN << "test_add_eq_1d_with_cpu succ" << RESET << std::endl;
    }
    ::free(cpu_res_buffer);
    ::free(gpu_res_buffer);
}

void test_gpu_add_eq_2d_with_cpu() {
    use_gpu(false);
    construct_env();
    int m = 103;
    int n = 80;
    Tensor *cpu_res = test_add_eq_2d_with_cpu_base(m, n);
    auto cpu_res_size = cpu_res->size();
    auto cpu_res_length = cpu_res->length();
    float *cpu_res_buffer = static_cast<float*>(::malloc(cpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(cpu_res_buffer),
        cpu_res,
        cpu_res_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    Tensor *gpu_res = test_add_eq_2d_with_cpu_base(m, n);
    auto gpu_res_size = gpu_res->size();
    auto gpu_res_length = gpu_res->length();
    float *gpu_res_buffer = static_cast<float*>(::malloc(gpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(gpu_res_buffer),
        gpu_res,
        gpu_res_size
    );
    destruct_env();
    assert(cpu_res_size == gpu_res_size);
    assert(cpu_res_length == gpu_res_length);
    const float eps = 1e-5f;
    //compare cpu and gpu result
    bool succ = true;
    for (int i = 0; i < cpu_res_length; ++ i) {
        if (fabs(cpu_res_buffer[i] - gpu_res_buffer[i]) > eps) {
            std::cerr << RED << "Error: cpu_res[" << i << "] = " << cpu_res_buffer[i]
                      << ", gpu_res[" << i << "] = " << gpu_res_buffer[i] << RESET << std::endl;
            succ = false;
            break;
        }
    }
    if (succ) {
        std::cout << GREEN << "test_add_eq_2d_with_cpu succ" << RESET << std::endl;
    }
    ::free(cpu_res_buffer);
    ::free(gpu_res_buffer);
}

Tensor *test_expand_add_with_cpu_base(int m, int n) {
    Tensor *input = allocTensor({m, n}, "input");
    Tensor *w = allocTensor({n}, "w");
    Tensor *res_wi_tensor = allocTensor({m, n}, "res_wi");
    gCreateAction(
        new ExpandAddAction(input, w, res_wi_tensor)
    );
    insert_boundary_action();
    allocMemAndInitTensors();
    input->fill(0.1f);
    std::vector<int> w_strides = w->get_strides();
    float *w_tmp_buffer = static_cast<float*>(::malloc(w->size()));
    auto shape = w->get_shape();
    for (int i = 0; i < shape[0]; ++ i) {
        w_tmp_buffer[i] = i;
    }
    g_backend_ops->cp_to_device(
        w,
        reinterpret_cast<char*>(w_tmp_buffer),
        w->size()
    );
    ::free(w_tmp_buffer);
    gDoActions();
    return res_wi_tensor;
}

void test_gpu_expand_add_with_cpu() {
    use_gpu(false);
    construct_env();
    int m = 103;
    int n = 80;
    Tensor *cpu_res = test_expand_add_with_cpu_base(m, n);
    auto cpu_res_size = cpu_res->size();
    auto cpu_res_length = cpu_res->length();
    float *cpu_res_buffer = static_cast<float*>(::malloc(cpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(cpu_res_buffer),
        cpu_res,
        cpu_res_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    Tensor *gpu_res = test_expand_add_with_cpu_base(m, n);
    auto gpu_res_size = gpu_res->size();
    auto gpu_res_length = gpu_res->length();
    float *gpu_res_buffer = static_cast<float*>(::malloc(gpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(gpu_res_buffer),
        gpu_res,
        gpu_res_size
    );
    destruct_env();
    assert(cpu_res_size == gpu_res_size);
    assert(cpu_res_length == gpu_res_length);
    const float eps = 1e-5f;
    //compare cpu and gpu result
    bool succ = true;
    for (int i = 0; i < cpu_res_length; ++ i) {
        if (fabs(cpu_res_buffer[i] - gpu_res_buffer[i]) > eps) {
            std::cerr << RED << "Error: cpu_res[" << i << "] = " << cpu_res_buffer[i]
                      << ", gpu_res[" << i << "] = " << gpu_res_buffer[i] << RESET << std::endl;
            succ = false;
            break;
        }
    }
    if (succ) {
        std::cout << GREEN << "test_expand_add_with_cpu succ" << RESET << std::endl;
    }
    ::free(cpu_res_buffer);
    ::free(gpu_res_buffer);
}

Tensor *test_mul_with_cpu_base(int m, int n) {
    Tensor *input = allocTensor({m, n}, "input");
    Tensor *w = allocTensor({m, n}, "w");
    Tensor *res_wi_tensor = allocTensor({m, n}, "res_wi");
    gCreateAction(
        new MulAction(input, w, res_wi_tensor)
    );
    insert_boundary_action();
    allocMemAndInitTensors();
    input->fill(0.1f);
    std::vector<int> w_strides = w->get_strides();
    float *w_tmp_buffer = static_cast<float*>(::malloc(w->size()));
    auto shape = w->get_shape();
    for (int i = 0; i < shape[0]; ++ i) {
        for (int j = 0; j < shape[1]; ++ j) {
            float *loc_w_tmp = w_tmp_buffer + i * w_strides[0] + j * w_strides[1];
            float v = i * shape[1] + j;
            *loc_w_tmp = v;
        }
    }
    g_backend_ops->cp_to_device(
        w,
        reinterpret_cast<char*>(w_tmp_buffer),
        w->size()
    );
    ::free(w_tmp_buffer);
    gDoActions();
    return res_wi_tensor;
}

void test_gpu_mul_with_cpu() {
    use_gpu(false);
    construct_env();
    int m = 103;
    int n = 80;
    Tensor *cpu_res = test_mul_with_cpu_base(m, n);
    auto cpu_res_size = cpu_res->size();
    auto cpu_res_length = cpu_res->length();
    float *cpu_res_buffer = static_cast<float*>(::malloc(cpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(cpu_res_buffer),
        cpu_res,
        cpu_res_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    Tensor *gpu_res = test_mul_with_cpu_base(m, n);
    auto gpu_res_size = gpu_res->size();
    auto gpu_res_length = gpu_res->length();
    float *gpu_res_buffer = static_cast<float*>(::malloc(gpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(gpu_res_buffer),
        gpu_res,
        gpu_res_size
    );
    destruct_env();
    assert(cpu_res_size == gpu_res_size);
    assert(cpu_res_length == gpu_res_length);
    const float eps = 1e-5f;
    //compare cpu and gpu result
    bool succ = true;
    for (int i = 0; i < cpu_res_length; ++ i) {
        if (fabs(cpu_res_buffer[i] - gpu_res_buffer[i]) > eps) {
            std::cerr << RED << "Error: cpu_res[" << i << "] = " << cpu_res_buffer[i]
                      << ", gpu_res[" << i << "] = " << gpu_res_buffer[i] << RESET << std::endl;
            succ = false;
            break;
        }
    }
    if (succ) {
        std::cout << GREEN << "test_mul_with_cpu succ" << RESET << std::endl;
    }
    ::free(cpu_res_buffer);
    ::free(gpu_res_buffer);
}

Tensor *test_gpu_sum_with_cpu_base(int m, int n) {
    Tensor *input = allocTensor({m, n}, "input");
    Tensor *res_wi_tensor = allocTensor({n}, "res_wi");
    gCreateAction(
        new SumAction(input, res_wi_tensor, 0)
    );
    insert_boundary_action();
    allocMemAndInitTensors();
    input->fill(0.1f);
    gDoActions();
    return res_wi_tensor;
}

void test_gpu_sum_with_cpu() {
    use_gpu(false);
    construct_env();
    int m = 103;
    int n = 80;
    Tensor *cpu_res = test_gpu_sum_with_cpu_base(m, n);
    auto cpu_res_size = cpu_res->size();
    auto cpu_res_length = cpu_res->length();
    float *cpu_res_buffer = static_cast<float*>(::malloc(cpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(cpu_res_buffer),
        cpu_res,
        cpu_res_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    Tensor *gpu_res = test_gpu_sum_with_cpu_base(m, n);
    auto gpu_res_size = gpu_res->size();
    auto gpu_res_length = gpu_res->length();
    float *gpu_res_buffer = static_cast<float*>(::malloc(gpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(gpu_res_buffer),
        gpu_res,
        gpu_res_size
    );
    destruct_env();
    assert(cpu_res_size == gpu_res_size);
    assert(cpu_res_length == gpu_res_length);
    const float eps = 1e-5f;
    //compare cpu and gpu result
    bool succ = true;
    for (int i = 0; i < cpu_res_length; ++ i) {
        if (fabs(cpu_res_buffer[i] - gpu_res_buffer[i]) > eps) {
            std::cerr << RED << "Error: cpu_res[" << i << "] = " << cpu_res_buffer[i]
                      << ", gpu_res[" << i << "] = " << gpu_res_buffer[i] << RESET << std::endl;
            succ = false;
            break;
        }
    }
    if (succ) {
        std::cout << GREEN << "test_sum_with_cpu succ" << RESET << std::endl;
    }
    ::free(cpu_res_buffer);
    ::free(gpu_res_buffer);
}

Tensor *test_cross_entropy_with_cpu_base(int m, int n) {
    Tensor *input = allocTensor({m, n}, "input");
    Tensor *labels = allocTensor({m}, "labels", INT32);
    Tensor *res_wi_tensor = allocTensor({1}, "res_wi");
    Tensor *sums = allocTensor({m}, "sums");
    Tensor *maxs = allocTensor({m}, "maxs");
    gCreateAction(
        new CrossEntropyAction(input, labels, sums, maxs, res_wi_tensor)
    );
    insert_boundary_action();
    allocMemAndInitTensors();
    input->fill(0.1f);
    gDoActions();
    return res_wi_tensor;
}

void test_gpu_cross_entropy_with_cpu() {
    use_gpu(false);
    construct_env();
    int m = 50;
    int n = 10;
    Tensor *cpu_res = test_cross_entropy_with_cpu_base(m, n);
    auto cpu_res_size = cpu_res->size();
    auto cpu_res_length = cpu_res->length();
    float *cpu_res_buffer = static_cast<float*>(::malloc(cpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(cpu_res_buffer),
        cpu_res,
        cpu_res_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    Tensor *gpu_res = test_cross_entropy_with_cpu_base(m, n);
    auto gpu_res_size = gpu_res->size();
    auto gpu_res_length = gpu_res->length();
    float *gpu_res_buffer = static_cast<float*>(::malloc(gpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(gpu_res_buffer),
        gpu_res,
        gpu_res_size
    );
    destruct_env();
    assert(cpu_res_size == gpu_res_size);
    assert(cpu_res_length == gpu_res_length);
    const float eps = 1e-3f;
    //compare cpu and gpu result
    bool succ = true;
    for (int i = 0; i < cpu_res_length; ++ i) {
        if (fabs(cpu_res_buffer[i] - gpu_res_buffer[i]) > eps) {
            std::cerr << std::setprecision(8) << RED << "Error: cpu_res[" << i << "] = " << cpu_res_buffer[i]
                      << ", gpu_res[" << i << "] = " << gpu_res_buffer[i] << RESET << std::endl;
            succ = false;
            break;
        }
    }
    if (succ) {
        std::cout << GREEN << "test_cross_entropy_with_cpu succ" << RESET << std::endl;
    }
    ::free(gpu_res_buffer);
    ::free(cpu_res_buffer);
}

Tensor *test_cross_entropy_backward_with_cpu_base(int m, int n) {
    Tensor *labels = allocTensor({m}, "input", INT32);
    Tensor *w = allocTensor({m, n}, "w");
    Tensor *res_wi_tensor = allocTensor({1}, "res_wi");
    Tensor *maxs_wi = allocTensor({m}, "maxs_wi");
    Tensor *sums_wi = allocTensor({m}, "sums_wi");
    Tensor *grad_wi = allocTensor({m, n}, "grad_wi");
    gCreateAction(
        new CrossEntropyAction(w, labels, maxs_wi, sums_wi, res_wi_tensor)
    );
    insert_boundary_action();
    gCreateAction(
        new CrossEntropyBackwardAction(w, labels, maxs_wi, sums_wi, grad_wi)
    );
    allocMemAndInitTensors();
    w->fill(0.1f);
    gDoActions();
    return grad_wi;
}

void test_gpu_cross_entropy_backward_with_cpu() {
    use_gpu(false);
    construct_env();
    int m = 103;
    int n = 10;
    Tensor *cpu_res = test_cross_entropy_backward_with_cpu_base(m, n);
    auto cpu_res_size = cpu_res->size();
    auto cpu_res_length = cpu_res->length();
    float *cpu_res_buffer = static_cast<float*>(::malloc(cpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(cpu_res_buffer),
        cpu_res,
        cpu_res_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    Tensor *gpu_res = test_cross_entropy_backward_with_cpu_base(m, n);
    auto gpu_res_size = gpu_res->size();
    auto gpu_res_length = gpu_res->length();
    float *gpu_res_buffer = static_cast<float*>(::malloc(gpu_res_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(gpu_res_buffer),
        gpu_res,
        gpu_res_size
    );
    destruct_env();
    assert(cpu_res_size == gpu_res_size);
    assert(cpu_res_length == gpu_res_length);
    const float eps = 1e-5f;
    //compare cpu and gpu result
    bool succ = true;
    for (int i = 0; i < cpu_res_length; ++ i) {
        if (fabs(cpu_res_buffer[i] - gpu_res_buffer[i]) > eps) {
            std::cerr << RED << "Error: cpu_res[" << i << "] = " << cpu_res_buffer[i]
                      << ", gpu_res[" << i << "] = " << gpu_res_buffer[i] << RESET << std::endl;
            succ = false;
            break;
        }
    }
    if (succ) {
        std::cout << GREEN << "test_cross_entropy_backward_with_cpu succ" << RESET << std::endl;
    }
    ::free(cpu_res_buffer);
    ::free(gpu_res_buffer);
}

void test_mlp_with_cpu_base(
     int m, int n, int k,
     int batch_size, int epochs,
     std::vector<float> &loss_res) {
    
    MLP mlp(
        m,
        {n, k},
        true
    );
    
    Tensor *input = allocTensor({batch_size, m}, "input");
    Tensor *labels = allocTensor({batch_size}, "labels", INT32);
    gCreateAction(
        new InitWeightAction(
            labels,
            "dbg",
            0,
            0
        )
    );
    auto n_input = graph::allocNode(input);
    n_input->init_weight_for_dbg();
    auto res = mlp.forward(n_input)->CrossEntropy(labels);
    zero_grad();
    insert_boundary_action();
    res->backward();
    Adam adam(
        mlp.get_parameters(),
        0.001f
    );
    adam.clip_grad(1.0f);
    adam.step();
    allocMemAndInitTensors();
    // printAllActions();
    float loss = 0;
    for (int i = 0; i < epochs; ++i) {
        gDoActions();
        float loss = 0;
        g_backend_ops->cp_from_device(
            reinterpret_cast<char*>(&loss),
            res->get_tensor(),
            sizeof(float)
        );
        loss_res.push_back(loss);
    }
}

void test_mlp_with_cpu() {
    int m = 784;
    int n = 30;
    int k = 10;
    int batch_size = 100;
    int epochs = 20;
    std::vector<float> loss_res_cpu;
    std::vector<float> loss_res_gpu;
    use_gpu(false);
    construct_env();
    test_mlp_with_cpu_base(m, n, k, batch_size, epochs, loss_res_cpu);
    destruct_env();
    // std::cout << "-------" << std::endl;
    use_gpu(true);
    construct_env();
    test_mlp_with_cpu_base(m, n, k, batch_size, epochs, loss_res_gpu);
    destruct_env();

    const float eps = 1e-2f;
    //compare cpu and gpu result
    bool succ = true;
    for (int i = 0; i < loss_res_cpu.size(); ++i) {
        float loss_cpu = loss_res_cpu[i] / batch_size;
        float loss_gpu = loss_res_gpu[i] / batch_size;
        if (fabs(loss_cpu - loss_gpu) > eps) {
            std::cerr << RED << std::setprecision(8) << "cpu_res[" << i << "] = " << loss_cpu
                      << ", gpu_res[" << i << "] = " << loss_gpu << RESET << std::endl;
            succ = false;
            break;
        }
    }
    if (succ) {
        std::cout << GREEN << "test_mlp_with_cpu succ" << RESET << std::endl;
    }
}

Tensor *test_repeat_interleave_with_cpu_base(int m, int n) {
    Tensor *input = allocTensor({m}, "input", INT32);
    auto node = graph::allocNode(input);
    node->init_weight_for_dbg();
    auto res = input->repeat_interleave(n);
    insert_boundary_action();
    // printAllActions();
    allocMemAndInitTensors();
    gDoActions();
    return res;
}

void test_repeat_interleave_with_cpu() {
    int m = 100;
    int n = 20;
    use_gpu(false);
    construct_env();
    auto res_cpu = test_repeat_interleave_with_cpu_base(m, n);
    auto res_cpu_size = res_cpu->size();
    auto res_cpu_length = res_cpu->length();
    int32_t *res_cpu_buffer = static_cast<int32_t*>(::malloc(res_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_cpu_buffer),
        res_cpu,
        res_cpu_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    auto res_gpu = test_repeat_interleave_with_cpu_base(m, n);
    auto res_gpu_size = res_gpu->size();
    auto res_gpu_length = res_gpu->length();
    int32_t *res_gpu_buffer = static_cast<int32_t*>(::malloc(res_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_gpu_buffer),
        res_gpu,
        res_gpu_size
    );
    destruct_env();

    assert(res_cpu_size == res_gpu_size);
    assert(res_cpu_length == res_gpu_length);
    bool succ = compare_ans1_ans2_int32(res_cpu_buffer, res_gpu_buffer, res_gpu_length);
    if (succ) {
        std::cout << GREEN << "test_repeat_interleave_with_gpu succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_repeat_interleave_with_gpu failed" << RESET << std::endl;
    }
    ::free(res_cpu_buffer);
    ::free(res_gpu_buffer);
}

Tensor *test_mask_with_cpu_base(int m, int n, int k) {
    Tensor *input = allocTensor({m, n, k}, "input");
    auto ni = graph::allocNode(input);
    ni->init_weight_for_dbg();
    Tensor *mask = allocTensor({m}, "mask", INT32);
    auto nm = graph::allocNode(mask);
    nm->init_weight_for_dbg();
    auto res = input->reshape({-1, k})->sequence_mask(mask->repeat_interleave(n), 0.1f);
    insert_boundary_action();
    allocMemAndInitTensors();
    gDoActions();
    return res;
}

void test_mask_with_cpu() {
    use_gpu(false);
    construct_env();
    int m = 100;
    int n = 500;
    int k = 30;
    auto res_cpu = test_mask_with_cpu_base(m, n, k);
    auto res_cpu_size = res_cpu->size();
    auto res_cpu_length = res_cpu->length();
    float *res_cpu_buffer = static_cast<float*>(::malloc(res_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_cpu_buffer),
        res_cpu,
        res_cpu_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    auto res_gpu = test_mask_with_cpu_base(m, n, k);
    auto res_gpu_size = res_gpu->size();
    auto res_gpu_length = res_gpu->length();
    float *res_gpu_buffer = static_cast<float*>(::malloc(res_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_gpu_buffer),
        res_gpu,
        res_gpu_size
    );
    destruct_env();
    assert(res_cpu_size == res_gpu_size);
    assert(res_cpu_length == res_gpu_length);
    bool succ = compare_ans1_ans2(res_cpu_buffer, res_gpu_buffer, res_gpu_length);
    if (succ) {
        std::cout << GREEN << "test_mask_with_gpu succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_mask_with_gpu failed" << RESET << std::endl;
    }
    ::free(res_gpu_buffer);
    ::free(res_cpu_buffer);
}

Tensor *test_mask_with_cpu_base_1(int m, int n, int k) {
    Tensor *input = allocTensor({m, n, k}, "input");
    auto ni = graph::allocNode(input);
    ni->init_weight_for_dbg();
    Tensor *mask = allocTensor({m*n}, "mask", INT32);
    auto nm = graph::allocNode(mask);
    nm->init_weight_for_dbg();
    auto res = input->reshape({-1, k})->sequence_mask(mask, 0.1f);
    insert_boundary_action();
    allocMemAndInitTensors();
    gDoActions();
    return res;
}

void test_mask_with_cpu_1() {
    use_gpu(false);
    construct_env();
    int m = 66;
    int n = 30;
    int k = 2000;
    auto res_cpu = test_mask_with_cpu_base_1(m, n, k);
    auto res_cpu_size = res_cpu->size();
    auto res_cpu_length = res_cpu->length();
    float *res_cpu_buffer = static_cast<float*>(::malloc(res_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_cpu_buffer),
        res_cpu,
        res_cpu_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    auto res_gpu = test_mask_with_cpu_base_1(m, n, k);
    auto res_gpu_size = res_gpu->size();
    auto res_gpu_length = res_gpu->length();
    float *res_gpu_buffer = static_cast<float*>(::malloc(res_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_gpu_buffer),
        res_gpu,
        res_gpu_size
    );
    destruct_env();
    assert(res_cpu_size == res_gpu_size);
    assert(res_cpu_length == res_gpu_length);
    bool succ = compare_ans1_ans2(res_cpu_buffer, res_gpu_buffer, res_gpu_length);
    if (succ) {
        std::cout << GREEN << "test_mask_with_gpu_1 succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_mask_with_gpu_1 failed" << RESET << std::endl;
    }
    ::free(res_gpu_buffer);
    ::free(res_cpu_buffer);
}

Tensor *test_softmax_with_cpu_base(int m, int n, int k) {
    Tensor *input = allocTensor({m, n, k}, "input");
    auto ni = graph::allocNode(input);
    ni->init_weight_for_dbg(10000.0f);
    auto res = input->softmax();
    insert_boundary_action();
    // printAllActions();
    allocMemAndInitTensors();
    gDoActions();
    std::vector<Tensor *> res_vec;
    return res;
}

void test_softmax_with_cpu() {
    use_gpu(false);
    construct_env();
    int m = 100;
    int n = 500;
    int k = 30;
    Tensor *res_cpu = test_softmax_with_cpu_base(m, n, k);
    auto res_cpu_size = res_cpu->size();
    auto res_cpu_length = res_cpu->length();
    float *res_cpu_buffer = static_cast<float*>(::malloc(res_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_cpu_buffer),
        res_cpu,
        res_cpu_size
    );
    destruct_env();

    use_gpu(true);
    construct_env();
    Tensor *res_gpu = test_softmax_with_cpu_base(m, n, k);
    auto res_gpu_size = res_gpu->size();
    auto res_gpu_length = res_gpu->length();
    float *res_gpu_buffer = static_cast<float*>(::malloc(res_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_gpu_buffer),
        res_gpu,
        res_gpu_size
    );
    destruct_env();

    assert(res_cpu_size == res_gpu_size);
    assert(res_cpu_length == res_gpu_length);
    bool succ_res = compare_ans1_ans2(res_cpu_buffer, res_gpu_buffer, res_gpu_length);
    if (!succ_res) {
        std::cerr << RED << "res mismatch" << RESET << std::endl;
    } else {
        std::cout << GREEN << "res succ" << RESET << std::endl;
    }
    
    bool succ = succ_res;
    if (succ) {
        std::cout << GREEN << "test_softmax_with_cpu succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_softmax_with_cpu failed" << RESET << std::endl;
    }
    ::free(res_cpu_buffer);
    ::free(res_gpu_buffer);
}

Tensor *test_masked_softmax_with_cpu_base(int m, int n, int k) {
    Tensor *input = allocTensor({m, n, k}, "input");
    auto ni = graph::allocNode(input);
    ni->init_weight_for_dbg(10000.0f);
    Tensor *valid_lens = allocTensor({m, n}, "mask", INT32);
    auto nm = graph::allocNode(valid_lens);
    nm->init_weight_for_dbg();
    auto res = ni->masked_softmax(valid_lens);
    insert_boundary_action();
    allocMemAndInitTensors();
    // printAllActions();
    gDoActions();
    return res->get_tensor();
}

void test_masked_softmax_with_cpu() {
    int m = 100;
    int n = 500;
    int k = 30;

    use_gpu(false);
    construct_env();
    auto res_cpu = test_masked_softmax_with_cpu_base(m, n, k);
    auto res_cpu_size = res_cpu->size();
    auto res_cpu_length = res_cpu->length();
    float *res_cpu_buffer = static_cast<float*>(::malloc(res_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_cpu_buffer),
        res_cpu,
        res_cpu_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    auto res_gpu = test_masked_softmax_with_cpu_base(m, n, k);
    auto res_gpu_size = res_gpu->size();
    auto res_gpu_length = res_gpu->length();
    float *res_gpu_buffer = static_cast<float*>(::malloc(res_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_gpu_buffer),
        res_gpu,
        res_gpu_size
    );
    destruct_env();
    assert(res_cpu_size == res_gpu_size);
    assert(res_cpu_length == res_gpu_length);
    bool succ = compare_ans1_ans2(res_cpu_buffer, res_gpu_buffer, res_gpu_length);
    if (succ) {
        std::cout << GREEN << "test_masked_softmax_with_cpu succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_masked_softmax_with_cpu failed" << RESET << std::endl;
    }
    ::free(res_gpu_buffer);
    ::free(res_cpu_buffer);
}

Tensor *test_masked_softmax_bp_with_cpu_base(
    int m, int n, int k
) {
    Tensor *input = allocTensor({m, n, k}, "input");
    auto ni = graph::allocNode(input);
    ni->require_grad();
    ni->init_weight_for_dbg(10000.0f);
    Tensor *valid_lens = allocTensor({m, n}, "mask", INT32);
    auto nm = graph::allocNode(valid_lens);
    nm->init_weight_for_dbg();
    Tensor *labels = allocTensor({m*n}, "input", INT32);
    auto nl = graph::allocNode(labels);
    nl->init_weight_for_dbg();
    auto res = ni->masked_softmax(valid_lens)->reshape({-1, k})->CrossEntropy(labels);
    insert_boundary_action();
    res->backward();
    // printAllActions();
    allocMemAndInitTensors();
    gDoActions();
    return ni->get_grad();
}

void test_masked_softmax_bp_with_cpu() {
    int m = 100;
    int n = 500;
    int k = 10;

    use_gpu(false);
    construct_env();
    auto res_cpu = test_masked_softmax_bp_with_cpu_base(m, n, k);
    auto res_cpu_size = res_cpu->size();
    auto res_cpu_length = res_cpu->length();
    float *res_cpu_buffer = static_cast<float*>(::malloc(res_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_cpu_buffer),
        res_cpu,
        res_cpu_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    auto res_gpu = test_masked_softmax_bp_with_cpu_base(m, n, k);
    auto res_gpu_size = res_gpu->size();
    auto res_gpu_length = res_gpu->length();
    float *res_gpu_buffer = static_cast<float*>(::malloc(res_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_gpu_buffer),
        res_gpu,
        res_gpu_size
    );
    destruct_env();
    assert(res_cpu_size == res_gpu_size);
    assert(res_cpu_length == res_gpu_length);

    bool succ = compare_ans1_ans2(res_cpu_buffer, res_gpu_buffer, res_gpu_length);
    if (succ) {
        std::cout << GREEN << "test_masked_softmax_bp_with_cpu succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_masked_softmax_bp_with_cpu failed" << RESET << std::endl;
    }

    ::free(res_gpu_buffer);
    ::free(res_cpu_buffer);
}

std::vector<Tensor *> test_bmm_bp_with_cpu_base(
    int batch, int m, int n, int k
) {
    Tensor *input = allocTensor({batch, m, n}, "input");
    auto ni = graph::allocNode(input);
    ni->require_grad();
    ni->init_weight_for_dbg(10000.0f);
    Tensor *w = allocTensor({batch, n, k}, "w");
    auto nw = graph::allocNode(w);
    nw->require_grad();
    nw->init_weight_for_dbg();
    Tensor *labels = allocTensor({batch * m}, "labels", INT32);
    auto nl = graph::allocNode(labels);
    nl->init_weight_for_dbg();
    auto res = ni->bmm(nw)->softmax();
    auto res_ce = res->reshape({-1, k})->CrossEntropy(labels);
    insert_boundary_action();
    res_ce->backward();
    allocMemAndInitTensors();
    gDoActions();
    return {res->get_tensor(), ni->get_grad(), nw->get_grad()};
}

void test_bmm_bp_with_cpu() {
    int batch = 32;
    int m = 50;
    int n = 512;
    int k = 10;

    use_gpu(false);
    construct_env();
    auto res_cpu_vec = test_bmm_bp_with_cpu_base(batch, m, n, k);
    auto res_cpu = res_cpu_vec[0];
    auto ni_grad_cpu = res_cpu_vec[1];
    auto nw_grad_cpu = res_cpu_vec[2];
    auto res_cpu_size = res_cpu->size();
    auto res_cpu_length = res_cpu->length();
    float *res_cpu_buffer = static_cast<float*>(::malloc(res_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_cpu_buffer),
        res_cpu,
        res_cpu_size
    );
    auto ni_grad_cpu_size = ni_grad_cpu->size();
    auto ni_grad_cpu_length = ni_grad_cpu->length();
    float *ni_grad_cpu_buffer = static_cast<float*>(::malloc(ni_grad_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(ni_grad_cpu_buffer),
        ni_grad_cpu,
        ni_grad_cpu_size
    );
    auto nw_grad_cpu_size = nw_grad_cpu->size();
    auto nw_grad_cpu_length = nw_grad_cpu->length();
    float *nw_grad_cpu_buffer = static_cast<float*>(::malloc(nw_grad_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nw_grad_cpu_buffer),
        nw_grad_cpu,
        nw_grad_cpu_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    auto res_gpu_vec = test_bmm_bp_with_cpu_base(batch, m, n, k);
    auto res_gpu = res_gpu_vec[0];
    auto ni_grad_gpu = res_gpu_vec[1];
    auto nw_grad_gpu = res_gpu_vec[2];
    auto res_gpu_size = res_gpu->size();
    auto res_gpu_length = res_gpu->length();
    float *res_gpu_buffer = static_cast<float*>(::malloc(res_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_gpu_buffer),
        res_gpu,
        res_gpu_size
    );
    auto ni_grad_gpu_size = ni_grad_gpu->size();
    auto ni_grad_gpu_length = ni_grad_gpu->length();
    float *ni_grad_gpu_buffer = static_cast<float*>(::malloc(ni_grad_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(ni_grad_gpu_buffer),
        ni_grad_gpu,
        ni_grad_gpu_size
    );
    auto nw_grad_gpu_size = nw_grad_gpu->size();
    auto nw_grad_gpu_length = nw_grad_gpu->length();
    float *nw_grad_gpu_buffer = static_cast<float*>(::malloc(nw_grad_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nw_grad_gpu_buffer),
        nw_grad_gpu,
        nw_grad_gpu_size
    );
    destruct_env();
    assert(res_cpu_size == res_gpu_size);
    assert(res_cpu_length == res_gpu_length);
    assert(ni_grad_cpu_size == ni_grad_gpu_size);
    assert(ni_grad_cpu_length == ni_grad_gpu_length);
    assert(nw_grad_cpu_size == nw_grad_gpu_size);
    assert(nw_grad_cpu_length == nw_grad_gpu_length);
    
    bool succ_res = compare_ans1_ans2(res_cpu_buffer, res_gpu_buffer, res_gpu_length, 1e-3);
    if (!succ_res) {
        std::cerr << RED << "res mismatch" << RESET << std::endl;
    }
    bool succ_ni_grad = compare_ans1_ans2(ni_grad_cpu_buffer, ni_grad_gpu_buffer, ni_grad_gpu_length, 1e-3);
    if (!succ_ni_grad) {
        std::cerr << RED << "ni_grad mismatch" << RESET << std::endl;
    }
    bool succ_nw_grad = compare_ans1_ans2(nw_grad_cpu_buffer, nw_grad_gpu_buffer, nw_grad_gpu_length, 1e-3);
    if (!succ_nw_grad) {
        std::cerr << RED << "nw_grad mismatch" << RESET << std::endl;
    }
    bool succ = succ_res && succ_ni_grad && succ_nw_grad;
    if (succ) {
        std::cout << GREEN << "test_bmm_bp_with_cpu succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_bmm_bp_with_cpu failed" << RESET << std::endl;
    }

    ::free(res_cpu_buffer);
    ::free(res_gpu_buffer);
    ::free(ni_grad_cpu_buffer);
    ::free(ni_grad_gpu_buffer);
    ::free(nw_grad_cpu_buffer);
    ::free(nw_grad_gpu_buffer);
}

std::vector<Tensor *> test_div_bp_with_cpu_base(
    int m, int n, int k
) {
    Tensor *input = allocTensor({m, n}, "input");
    auto ni = graph::allocNode(input);
    ni->require_grad();
    ni->init_weight_for_dbg(10000.0f);
    Tensor *w = allocTensor({n, k}, "w");
    auto nw = graph::allocNode(w);
    nw->require_grad();
    nw->init_weight_for_dbg(10000.0f);

    Tensor *labels = allocTensor({m}, "labels", INT32);
    auto n_labels = graph::allocNode(labels);
    n_labels->init_weight_for_dbg();

    auto i_shape = input->get_shape();
    auto w_shape = w->get_shape();

    auto res = ni->at(nw)->div(10.0f)->reshape({1, i_shape[0], w_shape[1]})->softmax()->reshape({i_shape[0], w_shape[1]});
    auto ce_res = res->CrossEntropy(labels);
    insert_boundary_action();
    ce_res->backward();

    allocMemAndInitTensors();
    gDoActions();

    std::vector<Tensor *> res_vec;
    res_vec.push_back(res->get_tensor());
    res_vec.push_back(ni->get_grad());
    res_vec.push_back(nw->get_grad());
    return res_vec;
}

void test_div_bp_with_cpu() {

    int m = 100;
    int n = 500;
    int k = 10;

    use_gpu(false);
    construct_env();
    auto res_cpu_vec = test_div_bp_with_cpu_base(m, n, k);
    auto res_cpu = res_cpu_vec[0];
    auto ni_grad_cpu = res_cpu_vec[1];
    auto nw_grad_cpu = res_cpu_vec[2];
    auto res_cpu_size = res_cpu->size();
    auto res_cpu_length = res_cpu->length();
    float *res_cpu_buffer = static_cast<float*>(::malloc(res_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_cpu_buffer),
        res_cpu,
        res_cpu_size
    );
    auto ni_grad_cpu_size = ni_grad_cpu->size();
    auto ni_grad_cpu_length = ni_grad_cpu->length();
    float *ni_grad_cpu_buffer = static_cast<float*>(::malloc(ni_grad_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(ni_grad_cpu_buffer),
        ni_grad_cpu,
        ni_grad_cpu_size
    );
    auto nw_grad_cpu_size = nw_grad_cpu->size();
    auto nw_grad_cpu_length = nw_grad_cpu->length();
    float *nw_grad_cpu_buffer = static_cast<float*>(::malloc(nw_grad_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nw_grad_cpu_buffer),
        nw_grad_cpu,
        nw_grad_cpu_size
    );
    destruct_env();

    use_gpu(true);
    construct_env();
    auto res_gpu_vec = test_div_bp_with_cpu_base(m, n, k);
    auto res_gpu = res_gpu_vec[0];
    auto ni_grad_gpu = res_gpu_vec[1];
    auto nw_grad_gpu = res_gpu_vec[2];
    
    auto res_gpu_size = res_gpu->size();
    auto res_gpu_length = res_gpu->length();
    float *res_gpu_buffer = static_cast<float*>(::malloc(res_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_gpu_buffer),
        res_gpu,
        res_gpu_size
    );
    
    auto ni_grad_gpu_size = ni_grad_gpu->size();
    auto ni_grad_gpu_length = ni_grad_gpu->length();
    float *ni_grad_gpu_buffer = static_cast<float*>(::malloc(ni_grad_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(ni_grad_gpu_buffer),
        ni_grad_gpu,
        ni_grad_gpu_size
    );
    auto nw_grad_gpu_size = nw_grad_gpu->size();
    auto nw_grad_gpu_length = nw_grad_gpu->length();
    float *nw_grad_gpu_buffer = static_cast<float*>(::malloc(nw_grad_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nw_grad_gpu_buffer),
        nw_grad_gpu,
        nw_grad_gpu_size
    );
    destruct_env();
    assert(res_cpu_size == res_gpu_size);
    assert(res_cpu_length == res_gpu_length);
    assert(ni_grad_cpu_size == ni_grad_gpu_size);
    assert(ni_grad_cpu_length == ni_grad_gpu_length);
    assert(nw_grad_cpu_size == nw_grad_gpu_size);
    assert(nw_grad_cpu_length == nw_grad_gpu_length);

    bool succ_res = compare_ans1_ans2(res_cpu_buffer, res_gpu_buffer, res_gpu_length);
    if (!succ_res) {
        std::cerr << RED << "res mismatch" << RESET << std::endl;
    }

    bool succ_ni_grad = compare_ans1_ans2(ni_grad_cpu_buffer, ni_grad_gpu_buffer, ni_grad_gpu_length);
    if (!succ_ni_grad) {
        std::cerr << RED << "ni_grad mismatch" << RESET << std::endl;
    }

    bool succ_nw_grad = compare_ans1_ans2(nw_grad_cpu_buffer, nw_grad_gpu_buffer, nw_grad_gpu_length);
    if (!succ_nw_grad) {
        std::cerr << RED << "nw_grad mismatch" << RESET << std::endl;
    }

    bool succ = succ_res && succ_ni_grad && succ_nw_grad;

    if (succ) {
        std::cout << GREEN << "test_div_bp_with_cpu succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_div_bp_with_cpu failed" << RESET << std::endl;
    }

    ::free(res_gpu_buffer);
    ::free(res_cpu_buffer);
    ::free(ni_grad_gpu_buffer);
    ::free(ni_grad_cpu_buffer);
    ::free(nw_grad_gpu_buffer);
    ::free(nw_grad_cpu_buffer);
}

std::vector<Tensor *> test_attention_bp_with_cpu_base(
    int batch, int m, int n, int k, int p
) {

    DotProductAttention attention;
    Tensor *querys = allocTensor({batch, m, n}, "querys");
    Tensor *keys = allocTensor({batch, k, n}, "keys");
    Tensor *values = allocTensor({batch, k, p}, "values");
    Tensor *valid_lens = allocTensor({batch}, "valid_lens", INT32);
    auto n_valied_lens = graph::allocNode(valid_lens);
    n_valied_lens->init_weight_for_dbg();
    Tensor *labels = allocTensor({batch*m}, "labels", INT32);
    auto n_labels = graph::allocNode(labels);
    n_labels->init_weight_for_dbg();
    auto nq = graph::allocNode(querys);
    nq->require_grad();
    nq->init_weight_for_dbg(1000000.0f);
    auto nk = graph::allocNode(keys);
    nk->require_grad();
    nk->init_weight_for_dbg(1000000.0f);
    auto nv = graph::allocNode(values);
    nv->require_grad();
    nv->init_weight_for_dbg(10000.0f);

    auto softmax_res = attention.forward(nq, nk, nv, valid_lens)->softmax();
    auto ce_res = softmax_res->reshape({-1, p})->CrossEntropy(labels);
    zero_grad();
    
    insert_boundary_action();
    ce_res->backward();
    // printAllActions();
    allocMemAndInitTensors();
    gDoActions();
    // std::cout << "nq grad: " << std::endl << *nq->get_grad() << std::endl;
    // std::cout << "softmax_res grad : " << std::endl << *softmax_res->get_grad() << std::endl;
    // std::cout << "nk grad: " << std::endl << *nk->get_grad() << std::endl;
    // std::cout << "nv grad: " << std::endl << *nv->get_grad() << std::endl;
    std::vector<Tensor *> res_vec;
    res_vec.push_back(softmax_res->get_tensor());
    res_vec.push_back(nq->get_grad());
    res_vec.push_back(nk->get_grad());
    res_vec.push_back(nv->get_grad());
    return res_vec;
}

void test_attention_bp_with_cpu() {

    int m = 100;
    int n = 400;
    int k = 512;
    int p = 10;
    int batch = 32;
    const float eps = 1e-2;

    // int m = 1;
    // int n = 1;
    // int k = 40;
    // int p = 10;
    // int batch = 1;
    use_gpu(false);
    construct_env();
    auto res_cpu_vec = test_attention_bp_with_cpu_base(batch, m, n, k, p);
    auto res_cpu = res_cpu_vec[0];
    auto nq_grad_cpu = res_cpu_vec[1];
    auto nk_grad_cpu = res_cpu_vec[2];
    auto nv_grad_cpu = res_cpu_vec[3];
    auto res_cpu_size = res_cpu->size();
    auto res_cpu_length = res_cpu->length();
    float *res_cpu_buffer = static_cast<float*>(::malloc(res_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_cpu_buffer),
        res_cpu,
        res_cpu_size
    );
    auto nq_grad_cpu_size = nq_grad_cpu->size();
    auto nq_grad_cpu_length = nq_grad_cpu->length();
    float *nq_grad_cpu_buffer = static_cast<float*>(::malloc(nq_grad_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nq_grad_cpu_buffer),
        nq_grad_cpu,
        nq_grad_cpu_size
    );
    // std::cout << "nq_grad_cpu : " << std::endl << *nq_grad_cpu << std::endl;
    auto nk_grad_cpu_size = nk_grad_cpu->size();
    auto nk_grad_cpu_length = nk_grad_cpu->length();
    float *nk_grad_cpu_buffer = static_cast<float*>(::malloc(nk_grad_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nk_grad_cpu_buffer),
        nk_grad_cpu,
        nk_grad_cpu_size
    );
    auto nv_grad_cpu_size = nv_grad_cpu->size();
    auto nv_grad_cpu_length = nv_grad_cpu->length();
    float *nv_grad_cpu_buffer = static_cast<float*>(::malloc(nv_grad_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nv_grad_cpu_buffer),
        nv_grad_cpu,
        nv_grad_cpu_size
    );
    destruct_env();
    use_gpu(true);
    construct_env();
    auto res_gpu_vec = test_attention_bp_with_cpu_base(batch, m, n, k, p);
    auto res_gpu = res_gpu_vec[0];
    auto nq_grad_gpu = res_gpu_vec[1];
    auto nk_grad_gpu = res_gpu_vec[2];
    auto nv_grad_gpu = res_gpu_vec[3];
    auto res_gpu_size = res_gpu->size();
    auto res_gpu_length = res_gpu->length();
    float *res_gpu_buffer = static_cast<float*>(::malloc(res_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_gpu_buffer),
        res_gpu,
        res_gpu_size
    );
    auto nq_grad_gpu_size = nq_grad_gpu->size();
    auto nq_grad_gpu_length = nq_grad_gpu->length();
    float *nq_grad_gpu_buffer = static_cast<float*>(::malloc(nq_grad_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nq_grad_gpu_buffer),
        nq_grad_gpu,
        nq_grad_gpu_size
    );
    // std::cout << "nq_grad_gpu : " << std::endl << *nq_grad_gpu << std::endl;
    auto nk_grad_gpu_size = nk_grad_gpu->size();
    auto nk_grad_gpu_length = nk_grad_gpu->length();
    float *nk_grad_gpu_buffer = static_cast<float*>(::malloc(nk_grad_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nk_grad_gpu_buffer),
        nk_grad_gpu,
        nk_grad_gpu_size
    );
    auto nv_grad_gpu_size = nv_grad_gpu->size();
    auto nv_grad_gpu_length = nv_grad_gpu->length();
    float *nv_grad_gpu_buffer = static_cast<float*>(::malloc(nv_grad_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(nv_grad_gpu_buffer),
        nv_grad_gpu,
        nv_grad_gpu_size
    );
    destruct_env();
    assert(res_cpu_size == res_gpu_size);
    assert(res_cpu_length == res_gpu_length);
    assert(nq_grad_cpu_size == nq_grad_gpu_size);
    assert(nq_grad_cpu_length == nq_grad_gpu_length);
    assert(nk_grad_cpu_size == nk_grad_gpu_size);
    assert(nk_grad_cpu_length == nk_grad_gpu_length);
    assert(nv_grad_cpu_size == nv_grad_gpu_size);
    assert(nv_grad_cpu_length == nv_grad_gpu_length);

    bool succ_res = compare_ans1_ans2(res_cpu_buffer, res_gpu_buffer, res_gpu_length, eps);
    if (!succ_res) {
        std::cerr << RED << "res mismatch" << RESET << std::endl;
    }
    bool succ_nq_grad = compare_ans1_ans2(nq_grad_cpu_buffer, nq_grad_gpu_buffer, nq_grad_gpu_length, eps);
    if (!succ_nq_grad) {
        std::cerr << RED << "nq_grad mismatch" << RESET << std::endl;
    }
    bool succ_nk_grad = compare_ans1_ans2(nk_grad_cpu_buffer, nk_grad_gpu_buffer, nk_grad_gpu_length, eps);
    if (!succ_nk_grad) {
        std::cerr << RED << "nk_grad mismatch" << RESET << std::endl;
    }
    bool succ_nv_grad = compare_ans1_ans2(nv_grad_cpu_buffer, nv_grad_gpu_buffer, nv_grad_gpu_length, eps);
    if (!succ_nv_grad) {
        std::cerr << RED << "nv_grad mismatch" << RESET << std::endl;
    }

    bool succ = succ_res && succ_nq_grad && succ_nk_grad && succ_nv_grad;
    if (succ) {
        std::cout << GREEN << "test_attention_bp_with_cpu succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_attention_bp_with_cpu failed" << RESET << std::endl;
    }

    ::free(res_gpu_buffer);
    ::free(res_cpu_buffer);
    ::free(nq_grad_gpu_buffer);
    ::free(nq_grad_cpu_buffer);
    ::free(nk_grad_gpu_buffer);
    ::free(nk_grad_cpu_buffer);
    ::free(nv_grad_gpu_buffer);
    ::free(nv_grad_cpu_buffer);
}

std::vector<Tensor *> test_permute_with_cpu_base(
    int m, int n, int k, int p, int q
) {
    Tensor *input = allocTensor({m, n, k, p}, "input");
    Tensor *w = allocTensor({p, q}, "w");
    auto ni = graph::allocNode(input);
    ni->require_grad();
    ni->init_weight_for_dbg();
    auto nw = graph::allocNode(w);
    auto res = ni->permute({2, 0, 1, 3})->reshape({-1, p})->at(nw);
    res->backward();
    insert_boundary_action();
    allocMemAndInitTensors();
    float *grad_buffer = static_cast<float*>(::malloc(m * n * k * q * sizeof(float)));
    assert(res->get_grad()->length() == m * n * k * q);
    for (int i = 0; i < res->get_grad()->length(); ++ i) {
       grad_buffer[i] = 1.0f;
    }
    g_backend_ops->cp_to_device(
        res->get_grad(),
        reinterpret_cast<char*>(grad_buffer),
        res->get_grad()->size()
    );
    ::free(grad_buffer);
    gDoActions();
    std::vector<Tensor *> res_vec;
    res_vec.push_back(res->get_tensor());
    res_vec.push_back(ni->get_grad());
    return res_vec;
}

void test_permute_with_cpu() {
    use_gpu(false);
    construct_env();
    int m = 10;
    int n = 50;
    int k = 30;
    int p = 10;
    int q = 20;

    auto res_cpu_vec = test_permute_with_cpu_base(m, n, k, p, q);
    auto res_cpu = res_cpu_vec[0];
    auto ni_grad_cpu = res_cpu_vec[1];
    auto res_cpu_size = res_cpu->size();
    auto res_cpu_length = res_cpu->length();
    float *res_cpu_buffer = static_cast<float*>(::malloc(res_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_cpu_buffer),
        res_cpu,
        res_cpu_size
    );
    auto ni_grad_cpu_size = ni_grad_cpu->size();
    auto ni_grad_cpu_length = ni_grad_cpu->length();
    float *ni_grad_cpu_buffer = static_cast<float*>(::malloc(ni_grad_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(ni_grad_cpu_buffer),
        ni_grad_cpu,
        ni_grad_cpu_size
    );
    destruct_env();

    use_gpu(true);
    construct_env();
    auto res_gpu_vec = test_permute_with_cpu_base(m, n, k, p, q);
    auto res_gpu = res_gpu_vec[0];
    auto ni_grad_gpu = res_gpu_vec[1];
    auto res_gpu_size = res_gpu->size();
    auto res_gpu_length = res_gpu->length();
    float *res_gpu_buffer = static_cast<float*>(::malloc(res_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_gpu_buffer),
        res_gpu,
        res_gpu_size
    );
    auto ni_grad_gpu_size = ni_grad_gpu->size();
    auto ni_grad_gpu_length = ni_grad_gpu->length();
    float *ni_grad_gpu_buffer = static_cast<float*>(::malloc(ni_grad_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(ni_grad_gpu_buffer),
        ni_grad_gpu,
        ni_grad_gpu_size
    );
    destruct_env();
    assert(res_cpu_size == res_gpu_size);
    assert(res_cpu_length == res_gpu_length);
    assert(ni_grad_cpu_size == ni_grad_gpu_size);
    assert(ni_grad_cpu_length == ni_grad_gpu_length);

    bool succ_res = compare_ans1_ans2(res_cpu_buffer, res_gpu_buffer, res_gpu_length);
    if (!succ_res) {
        std::cerr << RED << "res mismatch" << RESET << std::endl;
    }
    bool succ_ni_grad = compare_ans1_ans2(ni_grad_cpu_buffer, ni_grad_gpu_buffer, ni_grad_gpu_length);

    if (!succ_ni_grad) {
        std::cerr << RED << "ni_grad mismatch" << RESET << std::endl;
    }

    bool succ = succ_res && succ_ni_grad;
    if (succ) {
        std::cout << GREEN << "test_permute_with_cpu succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_permute_with_cpu failed" << RESET << std::endl;
    }
    ::free(res_gpu_buffer);
    ::free(res_cpu_buffer);
    ::free(ni_grad_gpu_buffer);
    ::free(ni_grad_cpu_buffer);
}

std::vector<Tensor *> test_embedding_with_cpu_base(int m, int n) {
    Embedding emb(m, n, true);
    Tensor *indices = allocTensor({m/2}, "indices", INT32);
    auto res = emb.forward(indices);
    insert_boundary_action();
    res->backward();
    allocMemAndInitTensors();
    int32_t *indices_buffer = static_cast<int32_t*>(::malloc(m/2 * sizeof(int32_t)));
    for (int i = 0; i < m/2; ++ i) {
        indices_buffer[i] = i*2;
    }
    g_backend_ops->cp_to_device(
        indices,
        reinterpret_cast<char*>(indices_buffer),
        indices->size()
    );
    ::free(indices_buffer);
    auto grad_length = res->get_grad()->length();
    assert(grad_length == m/2 * n);

    float *grad_buffer = static_cast<float*>(::malloc(grad_length * sizeof(float)));
    for (int i = 0; i < grad_length; ++ i) {
        grad_buffer[i] = 1.0f * i;
    }
    g_backend_ops->cp_to_device(
        res->get_grad(),
        reinterpret_cast<char*>(grad_buffer),
        res->get_grad()->size()
    );
    ::free(grad_buffer);

    gDoActions();
    std::vector<Tensor *> res_vec;
    res_vec.push_back(res->get_tensor());
    res_vec.push_back(emb.get_grad());
    return res_vec;
}

void test_embedding_with_cpu() {
    int m = 100;
    int n = 50;

    use_gpu(false);
    construct_env();
    auto res_cpu_vec = test_embedding_with_cpu_base(m, n);
    auto res_cpu = res_cpu_vec[0];
    auto emb_grad_cpu = res_cpu_vec[1];
    auto res_cpu_size = res_cpu->size();
    auto res_cpu_length = res_cpu->length();
    float *res_cpu_buffer = static_cast<float*>(::malloc(res_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_cpu_buffer),
        res_cpu,
        res_cpu_size
    );
    auto emb_grad_cpu_size = emb_grad_cpu->size();
    auto emb_grad_cpu_length = emb_grad_cpu->length();
    float *emb_grad_cpu_buffer = static_cast<float*>(::malloc(emb_grad_cpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(emb_grad_cpu_buffer),
        emb_grad_cpu,
        emb_grad_cpu_size
    );
    destruct_env();

    use_gpu(true);
    construct_env();
    auto res_gpu_vec = test_embedding_with_cpu_base(m, n);
    auto res_gpu = res_gpu_vec[0];
    auto emb_grad_gpu = res_gpu_vec[1];
    
    auto res_gpu_size = res_gpu->size();
    auto res_gpu_length = res_gpu->length();
    float *res_gpu_buffer = static_cast<float*>(::malloc(res_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(res_gpu_buffer),
        res_gpu,
        res_gpu_size
    );
    
    auto emb_grad_gpu_size = emb_grad_gpu->size();
    auto emb_grad_gpu_length = emb_grad_gpu->length();
    float *emb_grad_gpu_buffer = static_cast<float*>(::malloc(emb_grad_gpu_size));
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(emb_grad_gpu_buffer),
        emb_grad_gpu,
        emb_grad_gpu_size
    );
    destruct_env();
    
    assert(res_cpu_size == res_gpu_size);
    assert(res_cpu_length == res_gpu_length);
    assert(emb_grad_cpu_size == emb_grad_gpu_size);
    assert(emb_grad_cpu_length == emb_grad_gpu_length);
    bool succ_res = compare_ans1_ans2(res_cpu_buffer, res_gpu_buffer, res_gpu_length);
    if (!succ_res) {
        std::cerr << RED << "res mismatch" << RESET << std::endl;
    }

    bool succ_grad = compare_ans1_ans2(emb_grad_cpu_buffer, emb_grad_gpu_buffer, emb_grad_gpu_length);

    if (!succ_grad) {
        std::cerr << RED << "emb_grad mismatch" << RESET << std::endl;
    }

    bool succ = succ_res && succ_grad;
    if (succ) {
        std::cout << GREEN << "test_embedding_with_cpu succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_embedding_with_cpu failed" << RESET << std::endl;
    }

    ::free(res_gpu_buffer);
    ::free(res_cpu_buffer);
    ::free(emb_grad_gpu_buffer);
    ::free(emb_grad_cpu_buffer);
}

void test_gpu() {
    test_at();
    test_at_1();
    test_gpu_at_with_cpu();
    test_add();
    test_gpu_add_with_cpu();
    test_add_eq();
    test_gpu_add_eq_1d_with_cpu();
    test_gpu_add_eq_2d_with_cpu();
    test_expand_add();
    test_gpu_expand_add_with_cpu();
    test_mul();
    test_mul_1();
    test_gpu_mul_with_cpu();
    test_sum();
    test_gpu_sum_with_cpu();
    test_cross_entropy();
    test_gpu_cross_entropy_with_cpu();
    test_cross_entropy_backward();
    test_gpu_cross_entropy_backward_with_cpu();
    test_bp();
    test_adam();
    test_mlp();
    test_mlp_with_cpu();
    // test_print_tensor();
    test_contiguous();
    test_reshape();
    test_reshape_with_cpu();
    test_reshape_1();
    test_reshape_bp();
    test_reshape_bp_1();
    test_repeat_interleave();
    test_repeat_interleave_with_cpu();
    test_mask();
    test_mask_with_cpu();
    test_mask_1();
    test_mask_with_cpu_1();
    test_softmax();
    test_softmax_with_cpu();
    test_masked_softmax();
    test_masked_softmax_1();
    test_masked_softmax_with_cpu();
    test_masked_softmax_bp();
    test_masked_softmax_bp_with_cpu();
    test_bmm();
    test_bmm_1();
    test_bmm_2();
    test_bmm_bp();
    test_bmm_bp_with_cpu();
    test_div_bp();
    test_div_bp_with_cpu();
    test_bmm_bp_1();
    test_attention_bp();
    test_attention_bp_part();
    test_attention_bp_with_cpu();
    test_dropout();
    test_permute();
    test_permute_with_cpu();
    test_lazy_linear();
    test_mha();
    test_embedding();
    test_embedding_with_cpu();
    test_pe();
}

int main(int argc, char *argv[]) {
    int opt = 0;
    int backend_type = 1; // 0 is cpu 1 is gpu
    while ((opt = getopt(argc, argv, "t:")) != -1) {
        switch (opt) {
            case 't':
                backend_type = atoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -t <backend_type>" << std::endl;
                return 1;
        }
    }
    if (backend_type == 0) {
        test_cpu();
    } else if (backend_type == 1) {
        use_gpu();
        test_gpu();
    } else {
        std::cerr << "Invalid backend type. Use 0 for CPU and 1 for GPU." << std::endl;
        return 1;
    }
    return 0;
}
