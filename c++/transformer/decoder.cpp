#include "decoder.h"

DecoderBlock::DecoderBlock(
    uint _num_hidden,
    uint _ffn_num_hiddens,
    uint _num_heads,
    DATATYPE dropout,
    uint _index
)
    : num_hidden(_num_hidden),
    ffn_num_hiddens(_ffn_num_hiddens),
    num_heads(_num_heads),
    index(_index),
    training(true)
{
    self_attention = new MultiHeadAttention(num_heads, num_hidden, dropout);
    enc_attention = new MultiHeadAttention(num_heads, num_hidden, dropout);
    ffn = new PositionwiseFFN(ffn_num_hiddens, num_hidden);
    addnorm1 = new AddNorm(num_hidden, dropout);
    addnorm2 = new AddNorm(num_hidden, dropout);
    addnorm3 = new AddNorm(num_hidden, dropout);
}

DecoderBlock::~DecoderBlock() {
    delete self_attention;
    delete enc_attention;
    delete ffn;
    delete addnorm1;
    delete addnorm2;
    delete addnorm3;
}

void DecoderBlock::train(bool _training) {
    training = _training;
    self_attention->train(_training);
    enc_attention->train(_training);
    addnorm1->train(_training);
    addnorm2->train(_training);
    addnorm3->train(_training);
}

std::vector<autograd::Node *> DecoderBlock::forward(
    const std::vector<autograd::Node *> &X,
    const std::vector<autograd::Node *> &enc_outputs,
    DecoderContext *ctx
) {
    std::vector<autograd::Node *> key_values;
    if (ctx) { // predict
        assert(is_training() == false);
        assert(X.size() == ctx->dec_X.size());
        key_values.clear();
        key_values.reserve(X.size());
        for (size_t i = 0; i < X.size(); i++) {
            if (ctx->dec_X[i]) {
                std::vector<autograd::Node *> tmp = {ctx->dec_X[i], X[i]};
                ctx->dec_X[i] = autograd::cat(tmp, 0);
            } else {
                ctx->dec_X[i] = X[i];
            }
            key_values.push_back(ctx->dec_X[i]);
        }
    } else { // training
        assert(is_training());
        key_values = X;
    }

    std::vector<std::vector<uint>> valid_lens;
    valid_lens.clear();

    if (is_training()) {
        for (uint i = 0; i < X.size(); i++) {
            std::vector<uint> tmp;
            tmp.clear();
            auto q = X[i];
            Shape shape = q->getShape();
            for (uint j = 0; j < shape.rowCnt; j++) {
                tmp.push_back(j+1);
            }
            valid_lens.push_back(tmp);
        }
    }

    auto X2 = self_attention->forward(X, key_values, key_values, valid_lens);

}