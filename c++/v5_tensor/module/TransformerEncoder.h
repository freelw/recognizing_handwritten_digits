#ifndef TRANSFORMERENCODER_H
#define TRANSFORMERENCODER_H

#include "module/TransformerEncoderBlock.h"
#include "module/embedding.h"
#include "module/posencoding.h"

class TransformerEncoder {
    public:
        TransformerEncoder(
            int vocab_size, int num_hiddens, int ffn_num_hiddens,
            int num_heads, int num_blks, float dropout, bool bias = false
        );
        ~TransformerEncoder();
        graph::Node *forward(Tensor *indices, Tensor *valid_lens = nullptr);
        std::vector<Parameter *> get_parameters();
    private:
        int num_hiddens;
        Embedding *embedding;
        PosEncoding *pos_encoding;
        std::vector<TransformerEncoderBlock *> blks;
};


#endif