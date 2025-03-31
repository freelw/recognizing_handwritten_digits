#ifndef TRANSFORMER_SEQ2SEQ_H
#define TRANSFORMER_SEQ2SEQ_H

#include "encoder.h"
#include "decoder.h"
class Seq2SeqEncoderDecoder {
    public:
        Seq2SeqEncoderDecoder(
            Encoder *_encoder,
            Decoder *_decoder,
            uint _bos_id,
            uint _eos_id
        ) : encoder(_encoder), decoder(_decoder),
            bos_id(_bos_id), eos_id(_eos_id) {}
        ~Seq2SeqEncoderDecoder() {}
        autograd::Node* forward(
            const std::vector<std::vector<uint>> &src_token_ids,
            const std::vector<std::vector<uint>> &tgt_token_ids
        );
        std::vector<uint> predict(
            const std::vector<uint> &src_token_ids,
            uint max_len
        );
        std::vector<autograd::Parameters *> get_parameters();
        void train(bool _training) {
            encoder->train(_training);
            decoder->train(_training);
        }
    private:
        Encoder *encoder;
        Decoder *decoder;
        uint bos_id;
        uint eos_id;
};

#endif