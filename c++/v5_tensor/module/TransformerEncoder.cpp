#include "TransformerEncoder.h"
#include <cmath>

TransformerEncoder::TransformerEncoder(
    int vocab_size, int _num_hiddens, int ffn_num_hiddens,
    int num_heads, int num_blks, float dropout, bool bias
) : num_hiddens(_num_hiddens) {
    embedding = new Embedding(vocab_size, num_hiddens);
    pos_encoding = new PosEncoding(num_hiddens, dropout);
    for (int i = 0; i < num_blks; i++) {
        blks.push_back(new TransformerEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, bias));
    }   
}

TransformerEncoder::~TransformerEncoder() {
    delete embedding;
    delete pos_encoding;
    for (auto blk : blks) {
        delete blk;
    }
}

graph::Node *TransformerEncoder::forward(Tensor *indices, Tensor *valid_lens) {
    assert(indices->get_dim() == 2); // shape : (batch_size, seq_len)
    auto x = embedding->forward(indices->reshape({-1}))->mulsv(std::sqrt(num_hiddens));
    x = pos_encoding->forward(x);
    for (auto blk : blks) {
        x = blk->forward(x, valid_lens);
    }
    return x;
}