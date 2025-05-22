#ifndef TRANSFORMERENCODERBLOCK_H
#define TRANSFORMERENCODERBLOCK_H

#include "module/mha.h"
#include "module/PositionWiseFFN.h"
#include "module/AddNorm.h"

class TransformerEncoderBlock {
//num_hiddens, ffn_num_hiddens, num_heads, dropout,
//                 use_bias=False
    public:
        TransformerEncoderBlock(int num_hiddens, int ffn_num_hiddens, int num_heads,
                                float dropout, bool bias = false);
        ~TransformerEncoderBlock();
        graph::Node *forward(graph::Node *x, Tensor *valid_lens = nullptr);
        std::vector<Parameter *> get_parameters();
    private:
        MHA *attention;
        AddNorm *addnorm1;
        PositionWiseFFN *ffn;
        AddNorm *addnorm2;
};

#endif