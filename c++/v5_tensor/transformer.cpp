#include "module/Seq2Seq.h"
#include "common.h"
#include "optimizers/adam.h"
#include <unistd.h>

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
    // std::cout << "transformer parameters size : " << parameters.size() << std::endl;
    // for (int i = 0; i < parameters.size(); i++) {
    //     std::cout << "parameter " << i << " : " << parameters[i]->get_w()->get_meta_info() << std::endl;
    // }
}

void print_progress(const std::string &prefix, uint i, uint tot) {
    std::cout << "\r" << prefix << " [" << i << "/" << tot << "]" << std::flush;
}

void init_dec_valid_lens(Tensor *dec_valid_lens) {
    int32_t *dec_valid_lens_buffer = static_cast<int32_t *>(::malloc(
        dec_valid_lens->size()
    ));

    auto shape = dec_valid_lens->get_shape();

    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            dec_valid_lens_buffer[i * shape[1] + j] = j+1;
        }
    }

    g_backend_ops->cp_to_device(
        dec_valid_lens,
        reinterpret_cast<char*>(dec_valid_lens_buffer),
        dec_valid_lens->size()
    );

    ::free(dec_valid_lens_buffer);
}

int main(int argc, char *argv[]) {

    int opt;
    int epochs = 10;
    int batch_size = 128;
    int gpu = 1;
    float lr = 0.001f;

    while ((opt = getopt(argc, argv, "e:l:b:g:")) != -1) {
        switch (opt) {
            case 'e':
                epochs = atoi(optarg);
                break;
            case 'l':
                lr = atof(optarg);
                break;
            case 'b':
                batch_size = atoi(optarg);
                break;
            case 'g':
                gpu = atoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f <corpus> -c <checpoint> -e <epochs>" << std::endl;
                return 1;
        }
    }

    use_gpu(gpu==1);
    construct_env();
    int num_hiddens = 256;
    int num_blks = 2;
    float dropout = 0.2f;
    int ffn_num_hiddens = 64;
    int num_heads = 4;
    int enc_vocab_size = 1000; // fix me
    int dec_vocab_size = 1000;
    int bos_id = 0; // fix me
    int eos_id = 1; // fix me
    int num_steps = NUM_STEPS;
    int max_posencoding_len = MAX_POSENCODING_LEN;
    Seq2SeqEncoderDecoder *seq2seq = new Seq2SeqEncoderDecoder(
        bos_id, eos_id,
        enc_vocab_size, dec_vocab_size, num_hiddens, ffn_num_hiddens,
        num_heads, num_blks, max_posencoding_len, dropout
    );

    Tensor *src_token_ids = allocTensor({batch_size, num_steps}, INT32);
    Tensor *tgt_token_ids = allocTensor({batch_size, num_steps}, INT32);
    Tensor *enc_valid_lens = allocTensor({batch_size}, INT32);
    Tensor *dec_valid_lens = allocTensor({batch_size, num_steps}, INT32);
    Tensor *labels = allocTensor({batch_size * num_steps}, INT32);
    Tensor *ce_mask = allocTensor({batch_size * num_steps}, INT32);
    auto ce_mask_node = graph::allocNode(ce_mask);
    ce_mask_node->init_weight_fill(1.0f);
    auto res = seq2seq->forward(src_token_ids, tgt_token_ids, enc_valid_lens, dec_valid_lens);
    auto loss = res->reshape({-1, dec_vocab_size})->CrossEntropy(labels)->mask(ce_mask)->avg_1d(ce_mask);
    insert_boundary_action();
    
    std::vector<Parameter *> parameters = seq2seq->get_parameters();
    check_parameters(parameters, num_blks);
    Adam adam(parameters, 0.001);
    zero_grad();
    loss->backward();
    adam.clip_grad(1.0f);
    adam.step();
    // printAllActions();
    
    allocMemAndInitTensors();
    init_dec_valid_lens(dec_valid_lens);
    for (int i = 0; i < epochs; ++i) {
        for (int j = 0; j < 1306; ++j) {
            std::string prefix = "epoch " + std::to_string(i) + " : ";
            print_progress(prefix , j*batch_size, 1306*batch_size);
            gDoActions();
        }
        std::cout << "loss : " << *loss->get_tensor() << std::endl;
    }
    
    delete seq2seq;
    destruct_env();
    return 0;
}