#include "rnnlm.h"
#include "common.h"

#include <iostream>

RnnLM::RnnLM(Rnn *_rnn, uint _vocab_size, bool rand) : rnn(_rnn), vocab_size(_vocab_size) {
    if (!rand) {
        std::cerr << "Warning: using fixed weight for RnnLM" << std::endl;
    }
    fc = new Liner(rnn->get_hidden_num(), vocab_size, 1, rand);
}

RnnLM::~RnnLM() {
    delete fc;
}

Matrix * join_hiddens(const std::vector<Matrix *> &hiddens) {
    assert(hiddens.size() > 0);
    Matrix *res = allocTmpMatrix(Shape(hiddens[0]->getShape().rowCnt, hiddens.size()));
    for (uint i = 0; i < hiddens.size(); ++ i) {
        auto h = hiddens[i];
        h->checkShape(Shape(h->getShape().rowCnt, 1));
        for (uint j = 0; j < h->getShape().rowCnt; ++ j) {
            (*res)[j][i] = (*h)[j][0];
        }
    }
    return res;
}

Matrix *RnnLM::forward(RnnLMContext *ctx, const std::vector<Matrix *> &inputs) {
    RnnRes res = rnn->forward(ctx->rnn_ctx, inputs, nullptr);
    Matrix *hiddens = join_hiddens(res.states);
    return fc->forward(ctx->fc_ctx, hiddens);
}

void RnnLM::backward(RnnLMContext *ctx, Matrix* grad) {
    grad->checkShape(Shape(vocab_size, ctx->rnn_ctx->hiddens.size() - 1));
    Matrix *grad_hiddens = fc->backward(ctx->fc_ctx, grad);
    std::vector<Matrix *> grad_hiddens_vec = grad_hiddens->split(1);
    assert(grad_hiddens_vec.size() == ctx->rnn_ctx->hiddens.size() - 1);
    rnn->backward(ctx->rnn_ctx, grad_hiddens_vec);
}

RnnLMContext *RnnLM::init() {
    RnnLMContext *ctx = new RnnLMContext();
    ctx->rnn_ctx = rnn->init();
    ctx->fc_ctx = (LinerContext*)fc->init();
    return ctx;
}

void RnnLM::release(RnnLMContext *ctx) {
    rnn->release(ctx->rnn_ctx);
    fc->release(ctx->fc_ctx);
    delete ctx;
}

std::vector<Parameters*> RnnLM::get_parameters() {
    std::vector<Parameters*> res;
    auto rnn_params = rnn->get_parameters();
    res.insert(res.end(), rnn_params.begin(), rnn_params.end());
    auto fc_params = fc->get_parameters();
    res.insert(res.end(), fc_params.begin(), fc_params.end());
    return res;
}

void RnnLM::zero_grad() {
    rnn->zero_grad();
    fc->zero_grad();
}

void RnnLM::clip_grad(DATATYPE grad_clip_val) {
    std::vector<Parameters*> params = get_parameters();
    double norm = 0;
    for (auto param : params) {
        auto grad = param->get_grad();
        Shape shape = grad->getShape();
        for (uint i = 0; i < shape.rowCnt; ++ i) {
            for (uint j = 0; j < shape.colCnt; ++ j) {
                norm += std::pow((*grad)[i][j], 2);
            }
        }
    }
    norm = sqrt(norm);
    if (norm > grad_clip_val) {
        for (auto param : params) {
            // std::cout << "norm : " << norm << endl;
            auto grad = param->get_grad();
            *grad *= grad_clip_val / norm;
        }
    }
}

std::string RnnLM::predict(const std::string &prefix, uint num_preds) {
    std::vector<Matrix *> inputs;
    for (uint i = 0; i < prefix.size(); i++) {
        Matrix *m = allocTmpMatrix(Shape(vocab_size, 1));
        (*m)[to_index(prefix[i])][0] = 1;
        inputs.push_back(m);
    }
    RnnLMContext *ctx = init();
    RnnRes res = rnn->forward(ctx->rnn_ctx, inputs, nullptr);
    Matrix *last_hidden = res.states[res.states.size()-1];
    std::string ret = "";
    for (uint i = 0; i < num_preds; ++ i) {
        Matrix *output = fc->forward(ctx->fc_ctx, last_hidden);
        release(ctx);
        // std::cout << "output : " << *output << std::endl;
        uint max_index = 0;
        for (uint j = 0; j < vocab_size; ++ j) {
            if ((*output)[j][0] > (*output)[max_index][0]) {
                max_index = j;
            }
        }
        // std::cout << "max_index : " << max_index << std::endl;
        // std::cout << "to_char(max_index) : " << to_char(max_index) << std::endl;
        ret += to_char(max_index);
        Matrix *m = allocTmpMatrix(Shape(vocab_size, 1));
        (*m)[max_index][0] = 1;
        std::vector<Matrix *> new_inputs;
        new_inputs.push_back(m);
        ctx = init();
        res = rnn->forward(ctx->rnn_ctx, new_inputs, last_hidden);
        last_hidden = res.states[res.states.size()-1];
    }
    release(ctx);
    return ret;
}