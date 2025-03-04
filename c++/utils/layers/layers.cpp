#include "layers.h"

#include <cmath>
#include <vector>
#include <assert.h>
#include <random>
#include <chrono>
#include <iostream>

Liner::Liner(uint i, uint o, bool rand) : input_num(i), output_num(o) {
    weigt = new Parameters(Shape(o, i));
    bias = new Parameters(Shape(o, 1));

    // double stddev = sqrt(2./(input_num + output_num))*sqrt(2);
    double stddev = 0.02;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed);
    std::default_random_engine generator_b(seed+1024);
    std::normal_distribution<double> distribution_w(0.0, stddev);
    std::normal_distribution<double> distribution_b(0.0, 0.02);

    auto w = weigt->get_weight();
    auto b = bias->get_weight();
    for (uint i = 0; i < output_num; ++ i) {
        for (uint j = 0; j < input_num; ++ j) {
            (*w)[i][j] = rand ? distribution_w(generator_w) : 0.1;
        }
    }

    for (uint i = 0; i < output_num; ++ i) {
        (*b)[i][0] = rand ? distribution_b(generator_b) : 0.1;
    }
}

Liner::~Liner() {
    delete weigt;
    delete bias;   
}

Matrix *Liner::forward(Context *ctx, Matrix *input) {
    assert(input->getShape().rowCnt == input_num);
    LinerContext *ln_ctx = (LinerContext *)ctx;
    ln_ctx->input = input;
    auto w = weigt->get_weight();
    auto b = bias->get_weight();
    Matrix *res = w->dot(*input);
    
    for (uint j = 0; j < input->getShape().colCnt; ++ j) {
        for (uint i = 0; i < output_num; ++ i) {
            (*res)[i][j] += (*b)[i][0];
        }
    }
    return res;
}

Matrix *Liner::backward(Context *ctx, Matrix *grad) {
    assert(grad->getShape().rowCnt == output_num);
    LinerContext *ln_ctx = (LinerContext *)ctx;
    auto w = weigt->get_weight();
    Matrix *res_grad = w->transpose()->dot(*grad);
    Matrix *bias_grad = grad->sum(2);
    bias->set_grad(bias_grad);
    Matrix *weight_grad = grad->dot(*(ln_ctx->input->transpose()));
    weigt->set_grad(weight_grad);
    bias_grad->checkShape(*(bias->get_grad()));
    weight_grad->checkShape(*(weigt->get_grad()));
    return res_grad;
}

void Liner::zero_grad() {
    weigt->zero_grad();
    bias->zero_grad();
}

Context *Liner::init() {
    return new LinerContext();
}

void Liner::release(Context *ctx) {
    LinerContext *ln_ctx = (LinerContext *)ctx;
    delete ln_ctx;
}

std::vector<Parameters*> Liner::get_parameters() {
    std::vector<Parameters*> res;
    res.push_back(weigt);
    res.push_back(bias);
    return res;
}

Matrix *Relu::forward(Context *ctx, Matrix *input) {
    ReluContext *rl_ctx = (ReluContext*)ctx;
    rl_ctx->input = input;
    auto shape = input->getShape();
    Matrix *res = allocTmpMatrix(shape);
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            auto &value = (*input)[i][j];
            (*res)[i][j] = value > 0 ? value : 0;
        }
    }
    return res;
}

Matrix *Relu::backward(Context *ctx, Matrix *grad) {
    ReluContext *rl_ctx = (ReluContext*)ctx;
    auto &input = rl_ctx->input;
    input->checkShape(*grad);
    Matrix *res_grad = allocTmpMatrix(grad->getShape());
    auto shape = grad->getShape();
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            auto &value = (*input)[i][j];
            (*res_grad)[i][j] = value > 0 ? (*grad)[i][j] : 0;
        }
    }
    return res_grad;
}

Context *Relu::init() {
    return new ReluContext();
}

void Relu::release(Context * ctx) {
    ReluContext *rl_ctx = (ReluContext*)ctx;
    delete rl_ctx;
}

Matrix *CrossEntropyLoss::forward(Context * ctx, Matrix *input) {
    CrossEntropyLossContext *ce_ctx = (CrossEntropyLossContext*)ctx;
    ce_ctx->input = input;
    assert(input->getShape().colCnt == labels.size());
    Matrix *mExp = allocTmpMatrix(input);
    Matrix *loss = allocTmpMatrix(Shape(1,1));
    DATATYPE loss_value = 0;
    for (uint j = 0; j < input->getShape().colCnt; ++ j) {
        DATATYPE max = (*mExp)[0][j];
        for (uint i = 0; i < mExp->getShape().rowCnt; ++ i) {
            auto & e = (*mExp)[i][j];
            if (max < e) {
                max = e;
            }
        }
        DATATYPE sum = 0;
        for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
            DATATYPE & e = (*mExp)[i][j];
            e = std::exp(e-max);
            sum += e;
        }
        // assert(sum > 0);
        auto target = labels[j];
        DATATYPE ez = (*mExp)[target][j];
        CrosEntropyInfo p;
        p.sum = sum;
        p.max = max;
        ce_ctx->info.push_back(p);
        loss_value += -log(ez/sum);
    }
    // assert(ce_ctx->info.size() == labels.size());
    (*loss)[0][0] = loss_value/labels.size();
    return loss;
}

Matrix *CrossEntropyLoss::backward(Context *ctx, Matrix *) {
    CrossEntropyLossContext *ce_ctx = (CrossEntropyLossContext*)ctx;
    assert(ce_ctx->info.size() == labels.size());
    Matrix *grad = allocTmpMatrix(Shape(ce_ctx->input->getShape()));
    auto batch_size = labels.size();
    for (uint i = 0; i < batch_size; ++ i) {
        DATATYPE sum = ce_ctx->info[i].sum;
        DATATYPE max = ce_ctx->info[i].max;
        auto target = labels[i];
        for (uint j = 0; j < ce_ctx->input->getShape().rowCnt; ++j) {
            auto &_grad = (*grad)[j][i];
            _grad = std::exp((*ce_ctx->input)[j][i] - max) / sum / batch_size;
        }
        auto &_grad_target = (*grad)[target][i];
        _grad_target -= 1. / batch_size;
    }
    return grad;
}

Context *CrossEntropyLoss::init() {
    return new CrossEntropyLossContext();
}

void CrossEntropyLoss::release(Context *ctx) {
    CrossEntropyLossContext *ce_ctx = (CrossEntropyLossContext*)ctx;
    delete ce_ctx;
}

Rnn::Rnn(uint i, uint h, DATATYPE _sigma, bool _rand)
    : input_num(i), hidden_num(h), sigma(_sigma), rand(_rand) {
    wxh = new Parameters(Shape(hidden_num, input_num));
    whh = new Parameters(Shape(hidden_num, hidden_num));
    bh = new Parameters(Shape(hidden_num, 1));
    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed);
    std::default_random_engine generator_b(seed+1024);
    std::normal_distribution<DATATYPE> distribution_w(0.0, sigma);
    std::normal_distribution<DATATYPE> distribution_b(0.0, sigma);

    auto wxhw = wxh->get_weight();
    auto whhw = whh->get_weight();
    auto bhw = bh->get_weight();
    
    for (uint i = 0; i < hidden_num; ++ i) {
        for (uint j = 0; j < input_num; ++ j) {
            if (rand) {
                (*wxhw)[i][j] = distribution_w(generator_w);
            } else {
                (*wxhw)[i][j] = 0.1;
            }
        }
    }

    for (uint i = 0; i < hidden_num; ++ i) {
        if (rand) {
            (*bhw)[i][0] = distribution_b(generator_b);
        } else {
            (*bhw)[i][0] = 0.1;
        }
    }

    for (uint i = 0; i < hidden_num; ++ i) {
        for (uint j= 0; j < hidden_num; ++ j) {
            if (rand) {
                (*whhw)[i][j] = distribution_w(generator_w);
            } else {
                (*whhw)[i][j] = 0.1;
            }
        }
    }
}

Rnn::~Rnn() {
    delete wxh;
    delete whh;
    delete bh;
}

RnnRes Rnn::forward(RnnContext *ctx, const std::vector<Matrix *> &inputs, Matrix *hidden) {
    assert(inputs.size() >= 1);
    ctx->inputs = inputs;
    uint batch_size = inputs[0]->getShape().colCnt;

    if (!hidden) {
        hidden = allocTmpMatrix(Shape(hidden_num, batch_size));
    }
    ctx->hidden = hidden;
    hidden->checkShape(Shape(hidden_num, batch_size));

    RnnRes res;
    res.states.reserve(inputs.size());
    for (auto x : inputs) {
        Matrix *state = *(wxh->get_weight()->dot(*x)) + *(whh->get_weight()->dot(*hidden));
        Shape shape = state->getShape();
        for (uint i = 0; i < shape.rowCnt; ++ i) {
            for (uint j = 0; j < shape.colCnt; ++ j) {
                (*state)[i][j] += (*bh->get_weight())[i][0];
            }
        }
        hidden = state->tanh();
        res.states.push_back(hidden);
        ctx->hiddens.push_back(hidden);
    }
    return res;
}

Matrix *Rnn::backward(RnnContext *ctx, Matrix* grad) {
    grad->checkShape(Shape(hidden_num, 1));
    for (int i = ctx->inputs.size()-1; i >= 0; -- i) {
        auto x = ctx->inputs[i];
        auto h = ctx->hiddens[i];
        grad = grad->tanh_prime();
        bh->inc_grad(grad);
        Matrix *wxh_grad = grad->dot(*(x->transpose()));
        wxh->inc_grad(wxh_grad);
        Matrix *whh_grad = grad->dot(*(h->transpose()));
        whh->inc_grad(whh_grad);
        grad = whh->get_weight()->transpose()->dot(*grad);    
    }
    grad->checkShape(Shape(hidden_num, 1));
    return grad;
}

RnnContext *Rnn::init() {
    return new RnnContext();
}

void Rnn::release(RnnContext *ctx) {
    delete ctx;
}

std::vector<Parameters*> Rnn::get_parameters() {
    std::vector<Parameters*> res;
    res.push_back(wxh);
    res.push_back(whh);
    res.push_back(bh);
    return res;
}

void Rnn::zero_grad() {
    wxh->zero_grad();
    whh->zero_grad();
    bh->zero_grad();
}