#include "tensor.h"
#include "backends/cpu/cpu_ops.h"
#include "graph/node.h"
#include "optimizers/parameter.h"
#include "optimizers/adam.h"
#include "model/mlp.h"
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
        {30, 10}
    );
    Adam adam(
        mlp.get_parameters(),
        0.001f
    );

    Tensor *input = allocTensor({1, 784}, "input");
    Tensor *labels = allocTensor({1}, "labels", INT32);
    auto n_input = graph::allocNode(input);
    auto res = mlp.forward(n_input)->CrossEntropy(labels);
    zero_grad();
    insert_boundary_action();
    res->backward();
    adam.clip_grad(1.0f);
    adam.step();
    // printAllTensors();
    printAllActions();
    allocMemAndInitTensors();
    for (int i = 0; i < 500; ++ i) {
        gDoActions();
        float loss = 0;
        g_backend_ops->cp_from_device(
            reinterpret_cast<char*>(&loss),
            res->get_tensor(),
            sizeof(float)
        );
        std::cout << "loss : " << loss << std::endl;
    }

    auto w1_tensor = mlp.get_parameters()[0]->get_w();
    auto w2_tensor = mlp.get_parameters()[1]->get_w();

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

void test_cpu() {
    test_at();
    test_add();
    test_add_eq();
    test_expand_add();
    test_mul();
    test_sum();
    test_cross_entropy();
    test_cross_entropy_backward();
    test_bp();
    test_adam();
    test_mlp();
}

Tensor *test_add_with_cpu_base(int m, int n) {
    Tensor *input = allocTensor({m, n}, "input");
    Tensor *w = allocTensor({m, n}, "w");
    Tensor *res_wi_tensor = allocTensor({m, n}, "res_wi");
    gCreateAction(
        new AddAction(input, w, res_wi_tensor)
    );
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
}

Tensor *test_expand_add_with_cpu_base(int m, int n) {
    Tensor *input = allocTensor({m, n}, "input");
    Tensor *w = allocTensor({n}, "w");
    Tensor *res_wi_tensor = allocTensor({m, n}, "res_wi");
    gCreateAction(
        new ExpandAddAction(input, w, res_wi_tensor)
    );
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
}

Tensor *test_mul_with_cpu_base(int m, int n) {
    Tensor *input = allocTensor({m, n}, "input");
    Tensor *w = allocTensor({m, n}, "w");
    Tensor *res_wi_tensor = allocTensor({m, n}, "res_wi");
    gCreateAction(
        new MulAction(input, w, res_wi_tensor)
    );
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
}

Tensor *test_gpu_sum_with_cpu_base(int m, int n) {
    Tensor *input = allocTensor({m, n}, "input");
    Tensor *res_wi_tensor = allocTensor({n}, "res_wi");
    gCreateAction(
        new SumAction(input, res_wi_tensor, 0)
    );
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