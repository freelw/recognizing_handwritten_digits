#include "module/Seq2Seq.h"

int main(int argc, char *argv[]) {
    int num_hiddens = 256;
    int num_blks = 2;
    float dropout = 0.2f;
    int ffn_num_hiddens = 64;
    int num_heads = 4;
    int vocab_size = 1000; // fix me
    int bos_id = 0; // fix me
    int eos_id = 1; // fix me
    Seq2SeqEncoderDecoder *seq2seq = new Seq2SeqEncoderDecoder(
        bos_id, eos_id,
        vocab_size, num_hiddens, ffn_num_hiddens,
        num_heads, num_blks, dropout
    );

    delete seq2seq;
    return 0;
}