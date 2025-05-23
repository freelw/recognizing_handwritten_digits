#include "module/Seq2Seq.h"

void check_parameters(const std::vector<Parameter*> &parameters, int num_blks) {


    int parameters_size_should_be = 0;
    parameters_size_should_be += 1; // source embedding
    parameters_size_should_be += num_blks * (
        1 + // encoder block wq
        1 + // encoder block wk
        1 + // encoder block wv
        1 + // encoder block wo
        1 + // encoder addnorm1 gamma
        1 + // encoder addnorm1 beta
        1 + // encoder ffn w1
        1 + // encoder ffn b1
        1 + // encoder ffn w2
        1 + // encoder ffn b2
        1 + // encoder addnorm2 gamma
        1 // encoder addnorm2 beta
    );

    parameters_size_should_be += 1; // target embedding
    parameters_size_should_be += num_blks * (
        1 + // decoder block attention1 wq
        1 + // decoder block attention1 wk
        1 + // decoder block attention1 wv
        1 + // decoder block attention1 wo
        1 + // decoder block addnorm1 gamma
        1 + // decoder block addnorm1 beta
        1 + // decoder block attention2 wq
        1 + // decoder block attention2 wk
        1 + // decoder block attention2 wv
        1 + // decoder block attention2 wo
        1 + // decoder block addnorm2 gamma
        1 + // decoder block addnorm2 beta
        1 + // decoder block ffn w1
        1 + // decoder block ffn b1
        1 + // decoder block ffn w2
        1 + // decoder block ffn b2
        1 + // decoder block addnorm3 gamma
        1 // decoder block addnorm3 beta
    );
    parameters_size_should_be += 1; // target linear w
    parameters_size_should_be += 1; // target linear b
    assert(parameters.size() == parameters_size_should_be);
}

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

    std::vector<Parameter *> parameters = seq2seq->get_parameters();
    check_parameters(parameters, num_blks);

    delete seq2seq;
    return 0;
}