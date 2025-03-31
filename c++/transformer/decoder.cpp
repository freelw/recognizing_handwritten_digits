#include "decoder.h"
#include "macro.h"

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
    const std::vector<uint> &enc_valid_lens,
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
    std::vector<std::vector<uint>> dec_valid_lens;
    dec_valid_lens.clear();
    if (is_training()) {
        for (uint i = 0; i < X.size(); i++) {
            std::vector<uint> tmp;
            tmp.clear();
            auto q = X[i];
            Shape shape = q->getShape();
            for (uint j = 0; j < shape.rowCnt; j++) {
                tmp.push_back(j+1);
            }
            dec_valid_lens.push_back(tmp);
        }
    }
    auto X2 = self_attention->forward(X, key_values, key_values, dec_valid_lens);
    auto Y = addnorm1->forward(X, X2);
    auto Y2 = enc_attention->forward(Y, enc_outputs, enc_outputs, enc_valid_lens);
    auto Z = addnorm2->forward(Y, Y2);
    return addnorm3->forward(Z, ffn->forward(Z));
}

std::vector<autograd::Parameters *> DecoderBlock::get_parameters() {
    std::vector<autograd::Parameters *> res;
    std::vector<autograd::Parameters *> tmp;
    tmp = self_attention->get_parameters();
    res.insert(res.end(), tmp.begin(), tmp.end());
    tmp = enc_attention->get_parameters();
    res.insert(res.end(), tmp.begin(), tmp.end());
    tmp = ffn->get_parameters();
    res.insert(res.end(), tmp.begin(), tmp.end());
    tmp = addnorm1->get_parameters();
    res.insert(res.end(), tmp.begin(), tmp.end());
    tmp = addnorm2->get_parameters();
    res.insert(res.end(), tmp.begin(), tmp.end());
    tmp = addnorm3->get_parameters();
    res.insert(res.end(), tmp.begin(), tmp.end());
    return res;
}

Decoder::Decoder(
    uint _vocab_size,
    uint _num_hidden,
    uint _ffn_num_hiddens,
    uint _num_heads,
    uint _num_blks,
    DATATYPE dropout
)
    : vocab_size(_vocab_size),
    num_hidden(_num_hidden),
    ffn_num_hiddens(_ffn_num_hiddens),
    num_heads(_num_heads),
    num_blks(_num_blks),
    training(true)
{
    embedding = new autograd::Embedding(vocab_size, num_hidden);
    posencoding = new PosEncoding(MAX_POSENCODING_LEN, num_hidden, dropout);
    for (uint i = 0; i < num_blks; i++) {
        blocks.push_back(new DecoderBlock(num_hidden, ffn_num_hiddens, num_heads, dropout, i));
    }
    linear = new autograd::LazyLinear(vocab_size, true);
}

Decoder::~Decoder() {
    delete linear;
    for (auto blk : blocks) {
        delete blk;
    }
    delete posencoding;
    delete embedding;
}

void Decoder::train(bool _training) {
    training = _training;
    posencoding->train(_training);
    for (auto blk : blocks) {
        blk->train(_training);
    }
}

std::vector<autograd::Node *> Decoder::forward(
    const std::vector<std::vector<uint>> &inputs,
    const std::vector<autograd::Node *> &enc_outputs,
    const std::vector<uint> &enc_valid_lens,
    std::vector<autograd::Node *> &out_embs
) {
    auto embs = embedding->forward(inputs);
    out_embs = embs;
    std::vector<autograd::Node *> X;
    X.reserve(embs.size());
    for (auto & emb : embs) {
        X.push_back(emb->Mul(sqrt(num_hidden)));
    }
    X = posencoding->forward(X);
    DecoderContext *ctx = nullptr;
    if (is_training()) {
        ctx = new DecoderContext();
        for (uint i = 0; i < X.size(); i++) {
            ctx->dec_X.push_back(nullptr);
        }
    }
    for (auto blk : blocks) {
        X = blk->forward(X, enc_outputs, enc_valid_lens, ctx);
    }
    if (is_training()) {
        assert(ctx);
        delete ctx;
    }
    return linear->forward(X);
}