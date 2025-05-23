#ifndef TRANSFORMER_SEQ2SEQ_H
#define TRANSFORMER_SEQ2SEQ_H
#include "module/TransformerEncoder.h"
#include "module/TransformerDecoder.h"

class Seq2SeqEncoderDecoder {
    public:
        Seq2SeqEncoderDecoder(
            uint _bos_id,
            uint _eos_id,
            int vocab_size, int num_hiddens, int ffn_num_hiddens,
            int num_heads, int num_blks, float dropout, bool bias = false
        );
        ~Seq2SeqEncoderDecoder();
    private:
        TransformerEncoder *encoder;
        TransformerDecoder *decoder;
        uint bos_id;
        uint eos_id;
};

#endif