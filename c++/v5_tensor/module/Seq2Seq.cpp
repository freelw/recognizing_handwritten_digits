#include "Seq2Seq.h"

Seq2SeqEncoderDecoder::Seq2SeqEncoderDecoder(
    uint _bos_id,
    uint _eos_id,
    int vocab_size, int num_hiddens, int ffn_num_hiddens,
    int num_heads, int num_blks, float dropout, bool bias
) : bos_id(_bos_id), eos_id(_eos_id) {
    encoder = new TransformerEncoder(
        vocab_size, num_hiddens, ffn_num_hiddens,
        num_heads, num_blks, dropout, bias
    );
    decoder = new TransformerDecoder(
        vocab_size, num_hiddens, ffn_num_hiddens,
        num_heads, num_blks, dropout, bias
    );
}

Seq2SeqEncoderDecoder::~Seq2SeqEncoderDecoder() {
    assert(encoder != nullptr);
    assert(decoder != nullptr);
    delete encoder;
    delete decoder;
}


graph::Node * Seq2SeqEncoderDecoder::forward(
    Tensor *src_token_ids,
    Tensor *tgt_token_ids,
    Tensor *valid_lens
) {
    assert(src_token_ids->get_dim() == 2);
    assert(tgt_token_ids->get_dim() == 2);
    auto src_shape = src_token_ids->get_shape();
    auto tgt_shape = tgt_token_ids->get_shape();
    assert(src_shape == tgt_shape);
    assert(valid_lens->get_dim() == 1);
    assert(src_shape[0] == valid_lens->get_shape()[0]);
    assert(false);
}

std::vector<Parameter *> Seq2SeqEncoderDecoder::get_parameters() {
    std::vector<Parameter *> parameters;
    auto enc_parameters = encoder->get_parameters();
    auto dec_parameters = decoder->get_parameters();
    parameters.insert(parameters.end(), enc_parameters.begin(), enc_parameters.end());
    parameters.insert(parameters.end(), dec_parameters.begin(), dec_parameters.end());
    return parameters;
}