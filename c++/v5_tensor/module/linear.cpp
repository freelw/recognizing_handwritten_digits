#include "linear.h"
#include <cmath>

Linear::Linear(
    int input_num, int output_num,
    const std::string & prefix,
    float sigma,
    ACTIVATION act,
    bool _bias,
    bool const_weight
): bias(_bias) {
    auto w_tensor = allocTensor({input_num, output_num}, prefix + "_w_linear");
    w = graph::allocNode(w_tensor);
    w->require_grad();
    Pw = allocParameter(w);

    float mean = 0.0f;

    if (const_weight) {
        w->init_weight_for_dbg();
        return;
    } else {
        if (sigma <  0) {
            switch (act) {
                case RELU:
                    sigma = std::sqrt(2.0f/input_num);
                    break;
                case SIGMOID:
                case NONE:
                    sigma = std::sqrt(2.0f/(input_num + output_num));
                    break;
                case TANH:
                    sigma = std::sqrt(2.0 / (input_num + output_num)) * 4;
                    break;
                default:
                    assert(false);
            }
        }
        w->init_weight_gauss(sigma, mean);
    }
    if (bias) {
        auto b_tensor = allocTensor({output_num}, "_b_linear");
        b = graph::allocNode(b_tensor);
        b->require_grad();
        if (!const_weight) {
            b->init_weight_gauss(0.01f, 0.01f); // this is very important, break the symmetry
        }
        Pb = allocParameter(b);
    } else {
        b = nullptr;
    }
}

graph::Node *Linear::forward(graph::Node *input) {
    auto res = input->at(w);
    if (bias) {
        assert(b != nullptr);
        res = res->expand_add(b);
    }
    return res;
}

std::vector<Parameter *> Linear::get_parameters() {
    std::vector<Parameter *> params;
    params.push_back(Pw);
    if (bias) {
        assert(b != nullptr);
        assert(Pb != nullptr);
        params.push_back(Pb);
    }
    return params;
}

LazyLinear::~LazyLinear() {
    if (linear != nullptr) {
        delete linear;
    }
}

graph::Node *LazyLinear::forward(graph::Node *input) {
    if (linear == nullptr) {
        linear = new Linear(input_num, output_num, prefix, sigma, act, bias, const_weight);
    }
    return linear->forward(input);
}