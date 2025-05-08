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

    std::vector<int> w_strides = w->get_strides();
    std::vector<int> wt_strides = wt->get_strides();
    float *w_tmp_buffer = static_cast<float*>(::malloc(w->size()));
    float *wt_tmp_buffer = static_cast<float*>(::malloc(wt->size()));

    for (int i = 0; i < 3; ++ i) {
        for (int j = 0; j < 4; ++ j) {
            float *loc_w_tmp = w_tmp_buffer + i * w_strides[0] + j * w_strides[1];
            float *loc_wt_tmp = wt_tmp_buffer + j * wt_strides[0] + i * wt_strides[1];
            float v = i * 4 + j;
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
    gDoActions();
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

    
    const float eps = 1e-5f;
    bool succ = true;
    for (int i = 0; i < res_wi_tensor->length(); ++ i) {
        if (fabs(res_wi_tmp_buffer[i] - res_wti_tmp_buffer[i]) > eps) {
            succ = false;
            std::cerr << RED << "Error: res_wi[" << i << "] = " << res_wi_tmp_buffer[i]
                      << ", res_wti[" << i << "] = " << res_wti_tmp_buffer[i] << RESET << std::endl;
        }
    }
    if (succ) {
        std::cout << GREEN << "test_add succ" << RESET << std::endl;
    }

    ::free(res_wti_tmp_buffer);
    ::free(res_wi_tmp_buffer);
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
    auto input_data = static_cast<float*>(input->get_data());
    auto input1_data = static_cast<float*>(input1->get_data());
    const float eps = 1e-5f;
    bool succ = true;
    for (int i = 0; i < input->length(); ++ i) {
        if (fabs(input_data[i] - input1_data[i]) > eps) {
            succ = false;
            std::cerr << RED << "Error: res_wi[" << i << "] = " << input_data[i]
                      << ", res_wti[" << i << "] = " << input1_data[i] << RESET << std::endl;
        }
    }
    if (succ) {
        std::cout << GREEN << "test_add_eq succ" << RESET << std::endl;
    }

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
        std::cout << GREEN << "test_expand_add succ" << RESET << std::endl;
    }
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
        std::cout << GREEN << "test_mul succ" << RESET << std::endl;
    }
    destruct_env();
}

void test_sum() {
    construct_env();
    Tensor *w = allocTensor({3, 4}, "w");
    Tensor *wt = allocTensor({4, 3}, "wt");
    Tensor *res_wi_tensor = allocTensor({3, 4}, "res_wi");
    Tensor *res_wti_tensor = allocTensor({3, 4}, "res_wti");
    gCreateAction(
        new SumAction(w, res_wi_tensor, 0)
    );
    gCreateAction(
        new SumAction(wt->transpose(), res_wti_tensor, 0)
    );
    // printAllTensors();
    // printAllActions();
    allocMemAndInitTensors();
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
        std::cout << GREEN << "test_sum succ" << RESET << std::endl;
    }
    destruct_env();
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
    for (int i = 0; i < 3; ++ i) {
        int32_t *loc_labels = reinterpret_cast<int32_t*>(labels->location({i}));
        *loc_labels = i;
    }
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
        std::cout << GREEN << "test_cross_entropy succ" << RESET << std::endl;
    }
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
    for (int i = 0; i < 3; ++ i) {
        int32_t *loc_labels = reinterpret_cast<int32_t*>(labels->location({i}));
        *loc_labels = i;
    }
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

    auto grad_wi_data = static_cast<float*>(grad_wi->get_data());
    auto grad_wti_data = static_cast<float*>(grad_wti->get_data());

    const float eps = 1e-5f;
    bool succ = true;
    for (int i = 0; i < grad_wi->length(); ++ i) {
        if (fabs(grad_wi_data[i] - grad_wti_data[i]) > eps) {
            std::cerr << RED << "Error: grad_wi[" << i << "] = " << grad_wi_data[i]
                      << ", grad_wti[" << i << "] = " << grad_wti_data[i] << RESET << std::endl;
            succ = false;
        }
    }
    if (succ) {
        std::cout << GREEN << "test_cross_entropy_backward succ" << RESET << std::endl;
    }
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

    gDoActions();

    const float eps = 1e-5f;
    bool loss_succ = fabs(static_cast<float*>(nres->get_tensor()->get_data())[0] - 18.360287f) < eps;
    if (loss_succ) {
        std::cout << GREEN << "test_cross_entropy succ" << RESET << std::endl;
    } else {
        std::cout << RED << "test_cross_entropy failed" << RESET << std::endl;
    }

    auto nw_grad = nw->get_grad();
    auto nb_grad = nb->get_grad();
    auto nw1_grad = nw1->get_grad();
    auto nb1_grad = nb1->get_grad();

    bool nw_grad_succ = true;
    float nw_grad_ans[3][2] {
        17.997713,  19.797485,
        0.0000e+00,  0.0000e+00,
        -2.3890e-08, -2.6279e-08
    };
    for (int i = 0; i < nw_grad->get_shape()[0]; ++i) {
        for (int j = 0; j < nw_grad->get_shape()[1]; ++j) {
            float *loc_grad = static_cast<float*>(nw_grad->location({i, j}));
            if (fabs(*loc_grad - nw_grad_ans[i][j]) > eps) {
                std::cerr << std::setprecision(8) << RED << "Error: nw_grad[" << i << "][" << j << "] = " << *loc_grad
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
    
    for (int i = 0; i < nb_grad->get_shape()[0]; ++i) {
        float *loc_grad = static_cast<float*>(nb_grad->location({i}));
        if (fabs(*loc_grad - nb_grad_ans[i]) > eps) {
            std::cerr << std::setprecision(8) << RED << "Error: nb_grad[" << i << "] = " << *loc_grad
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

    bool nbw1_grad_succ = true;

    for (int i = 0; i < nw1_grad->get_shape()[0]; ++i) {
        for (int j = 0; j < nw1_grad->get_shape()[1]; ++j) {
            float *loc_grad = static_cast<float*>(nw1_grad->location({i, j}));
            if (fabs(*loc_grad - nw1_grad_ans[i][j]) > eps) {
                std::cerr << std::setprecision(8) << RED << "Error: nw1_grad[" << i << "][" << j << "] = " << *loc_grad
                          << ", nw1_grad_ans[" << i << "][" << j << "] = " << nw1_grad_ans[i][j] << RESET << std::endl;
                nbw1_grad_succ = false;
            }
        }
    }

    if (nbw1_grad_succ) {
        std::cout << GREEN << "test_cross_entropy nw1_grad succ" << RESET << std::endl;
    }

    float nb1_grad_ans[3] = {
        0.9997142,
        -1,
        0.00028578046
    };

    bool nb1_grad_succ = true;
    for (int i = 0; i < nb1_grad->get_shape()[0]; ++i) {
        float *loc_grad = static_cast<float*>(nb1_grad->location({i}));
        if (fabs(*loc_grad - nb1_grad_ans[i]) > eps) {
            std::cerr << std::setprecision(8) << RED << "Error: nb1_grad[" << i << "] = " << *loc_grad
                      << ", nb1_grad_ans[" << i << "] = " << nb1_grad_ans[i] << RESET << std::endl;
            nb1_grad_succ = false;
        }
    }

    if (nb1_grad_succ) {
        std::cout << GREEN << "test_cross_entropy nb1_grad succ" << RESET << std::endl;
    }

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
    Tensor *norm_before_clip = calc_norm(params);
    adam.clip_grad(1.0f);
    Tensor *norm_after_clip = calc_norm(params);
    adam.step();
    // printAllTensors();
    // printAllActions();
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

    gDoActions();

    auto nw_grad = nw->get_grad();
    auto nb_grad = nb->get_grad();
    auto nw1_grad = nw1->get_grad();
    auto nb1_grad = nb1->get_grad();

    const float eps = 1e-5f;
    bool nw_grad_succ = true;
    float nw_grad_ans[3][2] {
        0.5873974, 0.64613718,
        0, 0,
        -7.771136e-10, -8.5482493e-10,
    };
    for (int i = 0; i < nw_grad->get_shape()[0]; ++i) {
        for (int j = 0; j < nw_grad->get_shape()[1]; ++j) {
            float *loc_grad = static_cast<float*>(nw_grad->location({i, j}));
            if (fabs(*loc_grad - nw_grad_ans[i][j]) > eps) {
                std::cerr << std::setprecision(8) << RED << "Error: nw_grad[" << i << "][" << j << "] = " << *loc_grad
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
    
    for (int i = 0; i < nb_grad->get_shape()[0]; ++i) {
        float *loc_grad = static_cast<float*>(nb_grad->location({i}));
        if (fabs(*loc_grad - nb_grad_ans[i]) > eps) {
            std::cerr << std::setprecision(8) << RED << "Error: nb_grad[" << i << "] = " << *loc_grad
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

    bool nbw1_grad_succ = true;

    for (int i = 0; i < nw1_grad->get_shape()[0]; ++i) {
        for (int j = 0; j < nw1_grad->get_shape()[1]; ++j) {
            float *loc_grad = static_cast<float*>(nw1_grad->location({i, j}));
            if (fabs(*loc_grad - nw1_grad_ans[i][j]) > eps) {
                std::cerr << std::setprecision(8) << RED << "Error: nw1_grad[" << i << "][" << j << "] = " << *loc_grad
                          << ", nw1_grad_ans[" << i << "][" << j << "] = " << nw1_grad_ans[i][j] << RESET << std::endl;
                nbw1_grad_succ = false;
            }
        }
    }

    if (nbw1_grad_succ) {
        std::cout << GREEN << "test_adam clip nw1_grad succ" << RESET << std::endl;
    }

    float nb1_grad_ans[3] = {
        0.032628007,
        -0.032637335,
        9.3271128e-06
    };

    bool nb1_grad_succ = true;
    for (int i = 0; i < nb1_grad->get_shape()[0]; ++i) {
        float *loc_grad = static_cast<float*>(nb1_grad->location({i}));
        if (fabs(*loc_grad - nb1_grad_ans[i]) > eps) {
            std::cerr << std::setprecision(8) << RED << "Error: nb1_grad[" << i << "] = " << *loc_grad
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
            float *loc_w = static_cast<float*>(w->location({i, j}));
            if (fabs(*loc_w - w_ans[i][j]) > eps) {
                std::cerr << std::setprecision(8) << RED << "Error: w[" << i << "][" << j << "] = " << *loc_w
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
        float *loc_bias = static_cast<float*>(bias->location({i}));
        if (fabs(*loc_bias - bias_ans[i]) > eps) {
            std::cerr << std::setprecision(8) << RED << "Error: bias[" << i << "] = " << *loc_bias
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
            float *loc_w1 = static_cast<float*>(w1->location({i, j}));
            if (fabs(*loc_w1 - w1_ans[i][j]) > eps) {
                std::cerr << std::setprecision(8) << RED << "Error: w1[" << i << "][" << j << "] = " << *loc_w1
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
        float *loc_bias1 = static_cast<float*>(bias1->location({i}));
        if (fabs(*loc_bias1 - bias1_ans[i]) > eps) {
            std::cerr << std::setprecision(8) << RED << "Error: bias1[" << i << "] = " << *loc_bias1
                      << ", bias1_ans[" << i << "] = " << bias1_ans[i] << RESET << std::endl;
            bias1_succ = false;
        }
    }
    if (bias1_succ) {
        std::cout << GREEN << "test_adam bias1 succ" << RESET << std::endl;
    }

    destruct_env();
}

float calc_mean(Tensor *tensor) {
    float sum = 0.0f;
    auto data = static_cast<float*>(tensor->get_data());
    for (int i = 0; i < tensor->length(); ++i) {
        sum += data[i];
    }
    return sum / tensor->length();
}

float calc_std(Tensor *tensor) {
    float mean = calc_mean(tensor);
    float sum = 0.0f;
    auto data = static_cast<float*>(tensor->get_data());
    for (int i = 0; i < tensor->length(); ++i) {
        sum += (data[i] - mean) * (data[i] - mean);
    }
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

    Tensor *input = allocTensor({2, 784}, "input");
    Tensor *labels = allocTensor({2}, "labels", INT32);
    auto n_input = graph::allocNode(input);
    auto res = mlp.forward(n_input)->CrossEntropy(labels);
    zero_grad();
    res->backward();
    adam.clip_grad(1.0f);
    adam.step();
    // printAllTensors();
    // printAllActions();
    allocMemAndInitTensors();
    gDoActions();
    gDoActions();

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

void test_gpu() {
    test_add();
}

int main(int argc, char *argv[]) {
    int opt = 0;
    int backend_type = 0; // 0 is cpu 1 is gpu
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