#include "seq2seq.h"

autograd::Node* Seq2SeqEncoderDecoder::forward(
    const std::vector<std::vector<uint>> &src_token_ids,
    const std::vector<std::vector<uint>> &tgt_token_ids,
    const std::vector<uint> &valid_lens,
    std::vector<autograd::Node *> &enc_out_embs,
    std::vector<autograd::Node *> &dec_out_embs
) {
    auto hiddens = encoder->forward(src_token_ids, valid_lens, enc_out_embs);
    auto dec_outputs = decoder->forward(tgt_token_ids, hiddens, valid_lens, dec_out_embs);
    return autograd::cat(dec_outputs);
}

std::vector<autograd::Parameters *> Seq2SeqEncoderDecoder::get_parameters() {
    auto encoder_params = encoder->get_parameters();
    auto decoder_params = decoder->get_parameters();
    encoder_params.insert(encoder_params.end(), decoder_params.begin(), decoder_params.end());
    return encoder_params;
}