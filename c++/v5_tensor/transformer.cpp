#include "module/Seq2Seq.h"
#include "common.h"

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
    // print all parameters
    std::cout << "transformer parameters size : " << parameters.size() << std::endl;
    for (int i = 0; i < parameters.size(); i++) {
        std::cout << "parameter " << i << " : " << parameters[i]->get_w()->get_meta_info() << std::endl;
    }
}

int main(int argc, char *argv[]) {
    use_gpu();
    construct_env();
    int num_hiddens = 256;
    int num_blks = 2;
    float dropout = 0.2f;
    int ffn_num_hiddens = 64;
    int num_heads = 4;
    int vocab_size = 1000; // fix me
    int bos_id = 0; // fix me
    int eos_id = 1; // fix me
    int batch_size = 128;
    int num_steps = 9;
    int max_posencoding_len = MAX_POSENCODING_LEN;
    Seq2SeqEncoderDecoder *seq2seq = new Seq2SeqEncoderDecoder(
        bos_id, eos_id,
        vocab_size, num_hiddens, ffn_num_hiddens,
        num_heads, num_blks, max_posencoding_len, dropout
    );

    Tensor *src_token_ids = allocTensor({128, NUM_STEPS}, INT32);
    Tensor *tgt_token_ids = allocTensor({128, NUM_STEPS}, INT32);
    Tensor *enc_valid_lens = allocTensor({128}, INT32);
    Tensor *dec_valid_lens = allocTensor({128, NUM_STEPS}, INT32);
    auto res = seq2seq->forward(src_token_ids, tgt_token_ids, enc_valid_lens, dec_valid_lens);
    std::vector<Parameter *> parameters = seq2seq->get_parameters();
    check_parameters(parameters, num_blks);

    insert_boundary_action();
    allocMemAndInitTensors();
    gDoActions();

    delete seq2seq;
    destruct_env();
    return 0;
}