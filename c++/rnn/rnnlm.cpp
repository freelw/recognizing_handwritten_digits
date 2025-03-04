#include "rnnlm.h"

RnnLM::RnnLM(Rnn *_rnn, uint _vocab_size) : rnn(_rnn), vocab_size(_vocab_size) {
    fc = new Liner(rnn->get_hidden_num(), vocab_size, true);
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

Matrix *RnnLM::backward(RnnLMContext *ctx, Matrix* grad) {
    grad->checkShape(Shape(vocab_size, 1));
    Matrix *grad_hiddens = fc->backward(ctx->fc_ctx, grad);
    grad_hiddens->checkShape(Shape(rnn->get_hidden_num(), 1));
    Matrix *res = rnn->backward(ctx->rnn_ctx, grad_hiddens);
    return res;
}

RnnLMContext *RnnLM::init() {
    RnnLMContext *ctx = new RnnLMContext();
    ctx->rnn_ctx = rnn->init();
    ctx->fc_ctx = (LinerContext*)fc->init();
    return ctx;
}

void RnnLM::release(RnnContext *ctx) {
    RnnLMContext *lm_ctx = (RnnLMContext *)ctx;
    rnn->release(lm_ctx->rnn_ctx);
    fc->release(lm_ctx->fc_ctx);
    delete lm_ctx;
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