#include "TransformerDecoder.h"
#include <cmath>

TransformerDecoder::TransformerDecoder(
    int vocab_size,
    int _num_hiddens,
    int ffn_num_hiddens,
    int num_heads,
    int num_blks,
    float dropout
) : num_hiddens(_num_hiddens) {
    embedding = new Embedding(vocab_size, num_hiddens);
    pos_encoding = new PosEncoding(num_hiddens, dropout);
    
    for (int i = 0; i < num_blks; i++) {
        blks.push_back(new TransformerDecoderBlock(
            num_hiddens, ffn_num_hiddens, num_heads, dropout
        ));
    }
    dense = new LazyLinear(vocab_size, "dense", -1.0f, -1.0f, NONE);
}

TransformerDecoder::~TransformerDecoder() {
    assert(embedding != nullptr);
    assert(pos_encoding != nullptr);
    assert(dense != nullptr);
    for (auto blk : blks) {
        assert(blk != nullptr);
        delete blk;
    }
    delete embedding;
    delete pos_encoding;
    delete dense;
}

graph::Node *TransformerDecoder::forward(
    Tensor *tgt_token_ids,
    graph::Node *enc_outputs,
    Tensor *enc_valid_lens,
    Tensor *dec_valid_lens
) {
    auto x = embedding->forward(tgt_token_ids)->mulsv(std::sqrt(num_hiddens));
    x = pos_encoding->forward(x);
    for (auto blk : blks) {
        x = blk->forward(x, enc_outputs, enc_valid_lens, dec_valid_lens);
    }
    return dense->forward(x);
}

