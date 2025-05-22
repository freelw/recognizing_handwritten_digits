#ifndef TRANSFORMERDECODERBLOCK_H
#define TRANSFORMERDECODERBLOCK_H

#include "module/mha.h"
#include "module/PositionWiseFFN.h"
#include "module/AddNorm.h"

class TransformerDecoderBlock {
    public:
        TransformerDecoderBlock(int num_hiddens, int ffn_num_hiddens, int num_heads,
                                float dropout, bool bias = false);
        ~TransformerDecoderBlock();
        graph::Node *forward(graph::Node *x, graph::Node *state, Tensor *valid_lens = nullptr);
        std::vector<Parameter *> get_parameters();
    private:
        MHA *masked_attention;
        MHA *attention;
        AddNorm *addnorm1;
        PositionWiseFFN *ffn;
        AddNorm *addnorm2;

};

#endif