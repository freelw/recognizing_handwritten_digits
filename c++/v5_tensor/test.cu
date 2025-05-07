#include "tensor.h"
#include "backends/cpu/cpu_ops.h"
#include "graph/node.h"
#include "common.h"
#include <iomanip>

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
    sanitizeTensors();
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

void test_add_eq() {
    init_backend();
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

    sanitizeTensors();

    freeAllActions();
    freeAllTensors();
    releaseTensorMem();
    release_backend();
}

void test_expand_add() {

    init_backend();
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

    sanitizeTensors();

    freeAllActions();
    freeAllTensors();
    releaseTensorMem();
    release_backend();
}

void test_mul() {
    init_backend();
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

    sanitizeTensors();

    freeAllActions();
    freeAllTensors();
    releaseTensorMem();
    release_backend();
}

void test_sum() {
    init_backend();
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

    sanitizeTensors();

    freeAllActions();
    freeAllTensors();
    releaseTensorMem();
    release_backend();
}

void test_cross_entropy() {
    init_backend();
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

    sanitizeTensors();

    freeAllActions();
    freeAllTensors();
    releaseTensorMem();
    release_backend();
}

void test_cross_entropy_backward() {
    init_backend();
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
    sanitizeTensors();
    freeAllActions();
    freeAllTensors();
    releaseTensorMem();
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

    // // print forward result
    // std::cout << "forward result0: " << std::endl;
    // for (int i = 0; i < foward_res0->get_tensor()->length(); ++i) {
    //     std::cout << static_cast<float*>(foward_res0->get_tensor()->get_data())[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "forward result1: " << std::endl;
    // for (int i = 0; i < foward_res1->get_tensor()->length(); ++i) {
    //     std::cout << static_cast<float*>(foward_res1->get_tensor()->get_data())[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "loss : " << std::setprecision(8) << static_cast<float*>(nres->get_tensor()->get_data())[0] << std::endl;
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

    // print gradient
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

    // print nb_grad

    // for (int i = 0; i < nb_grad->get_shape()[0]; ++i) {
    //     float *loc_grad = static_cast<float*>(nb_grad->location({i}));
    //     std::cout << std::setprecision(8) << "nb_grad[" << i << "] = " << *loc_grad << std::endl;
    // }
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

    sanitizeTensors();
    freeAllActions();
    freeAllTensors();
    releaseTensorMem();
    release_backend();
}

void test() {
    test_at();
    test_add();
    test_add_eq();
    test_expand_add();
    test_mul();
    test_sum();
    test_cross_entropy();
    test_cross_entropy_backward();
    test_bp();
}

int main() {
    test();
    freeAllTensors();
    freeAllTensorViews();
    graph::freeAllEdges();
    graph::freeAllNodes();
    freeAllActions();
    return 0;
}