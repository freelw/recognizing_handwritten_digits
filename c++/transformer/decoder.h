#ifndef DECODER_H
#define DECODER_H


#include "autograd/node.h"
#include "attention.h"
#include "addnorm.h"
#include "ffn.h"

struct DecoderContext {
    std::vector<autograd::Node *> dec_X;
};

class DecoderBlock {
    public:
        DecoderBlock(
            uint _num_hidden,
            uint _ffn_num_hiddens,
            uint _num_heads,
            DATATYPE dropout,
            uint _index
        );
        ~DecoderBlock();
        bool is_training() { return training; }
        void train(bool _training);
        std::vector<autograd::Node *> forward(
            const std::vector<autograd::Node *> &X,
            const std::vector<autograd::Node *> &enc_outputs,
            const std::vector<uint> &valid_lens,
            DecoderContext *ctx
        );
    private:
        uint num_hidden;
        uint ffn_num_hiddens;
        uint num_heads;
        uint index;
        bool training;
        MultiHeadAttention *self_attention;
        MultiHeadAttention *enc_attention;
        PositionwiseFFN *ffn;
        AddNorm *addnorm1;
        AddNorm *addnorm2;
        AddNorm *addnorm3;
};
#endif