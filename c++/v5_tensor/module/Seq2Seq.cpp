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