#include "common.h"
#include "checkpoint.h"
#include "dataloader.h"
#include "module/Seq2Seq.h"
#include "optimizers/adam.h"
#include <unistd.h>
#include <iomanip>
#include <signal.h>

extern bool shutdown;
void signal_callback_handler(int signum);

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

std::vector<uint> trim_or_padding(const std::vector<uint> &src, uint max_len, uint pad_id) {
    std::vector<uint> res = src;
    if (src.size() > max_len) {
        res.resize(max_len);
    } else {
        res.resize(max_len, pad_id);
    }
    return res;
}

std::vector<uint> add_bos(const std::vector<uint> &src, uint bos_id) {
    std::vector<uint> res = src;
    res.insert(res.begin(), bos_id);
    return res;
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

void load_tokens_from_file(
    const std::string & corpus,
    std::vector<std::vector<uint>> &src_token_ids,
    std::vector<std::vector<uint>> &tgt_token_ids,
    int &enc_vocab_size,
    int &dec_vocab_size,
    int &bos_id,
    int &eos_id,
    int &src_pad_id,
    int &tgt_pad_id
    ) {
    std::string src_vocab_name = SRC_VOCAB_NAME;
    std::string tgt_vocab_name = TGT_VOCAB_NAME;
    std::string test_file = TEST_FILE;
    seq2seq::DataLoader loader(corpus, src_vocab_name, tgt_vocab_name, test_file);
    loader.get_token_ids(src_token_ids, tgt_token_ids);
    enc_vocab_size = loader.src_vocab_size();
    dec_vocab_size = loader.tgt_vocab_size();
    bos_id = loader.tgt_bos_id();
    eos_id = loader.tgt_eos_id();
    src_pad_id = loader.src_pad_id();
    tgt_pad_id = loader.tgt_pad_id();
}

std::string generateDateTimeSuffix() {    
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    struct std::tm* localTime = std::localtime(&currentTime);
    std::ostringstream oss;
    oss << std::put_time(localTime, "_%Y%m%d_%H%M%S");
    return oss.str();
}

int main(int argc, char *argv[]) {

    signal(SIGINT, signal_callback_handler);
    shutdown = false;

    int opt;
    int epochs = 10;
    int batch_size = 128;
    int gpu = 1;
    float lr = 0.001f;
    std::string checkpoint;
    std::string corpus = RESOURCE_NAME;

    while ((opt = getopt(argc, argv, "f:c:e:l:b:g:")) != -1) {
        switch (opt) {
            case 'f':
                corpus = optarg;
                break;
            case 'c':
                checkpoint = optarg;
                break;
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
                std::cerr << "Usage: " << argv[0] 
                    << " -f <corpus> -c <checpoint> -e <epochs>" << std::endl;
                return 1;
        }
    }

    std::cout << "corpus : " << corpus << std::endl;
    std::cout << "epochs : " << epochs << std::endl;
    std::cout << "batch_size : " << batch_size << std::endl;
    std::cout << "gpu : " << gpu << std::endl;
    std::cout << "learning rate : " << lr << std::endl;
    std::cout << "checkpoint : " << checkpoint << std::endl;

    int enc_vocab_size = 0;
    int dec_vocab_size = 0;
    int bos_id = 0;
    int eos_id = 0;
    int src_pad_id = 0;
    int tgt_pad_id = 0;

    std::vector<std::vector<uint>> v_src_token_ids;
    std::vector<std::vector<uint>> v_tgt_token_ids;
    load_tokens_from_file(
        corpus,
        v_src_token_ids, v_tgt_token_ids,
        enc_vocab_size, dec_vocab_size,
        bos_id,
        eos_id,
        src_pad_id,
        tgt_pad_id
    );
    std::cout << "enc_vocab_size : " << enc_vocab_size << std::endl;
    std::cout << "dec_vocab_size : " << dec_vocab_size << std::endl;
    std::cout << "bos_id : " << bos_id << std::endl;
    std::cout << "eos_id : " << eos_id << std::endl;
    std::cout << "src_pad_id : " << src_pad_id << std::endl;
    std::cout << "tgt_pad_id : " << tgt_pad_id << std::endl;

    use_gpu(gpu==1);
    construct_env();
    zero_c_tensors();
    zero_grad();
    int num_hiddens = 256;
    int num_blks = 2;
    float dropout = 0.2f;
    int ffn_num_hiddens = 64;
    int num_heads = 4;
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
    Tensor *ce_mask = allocTensor({batch_size * num_steps});

    // alloc input buffers
    // 1. enc_valid_lens
    // 2. src_token_ids
    // 3. tgt_token_ids
    // 4. labels
    // 5. ce_mask
    // 6. dec_valid_lens 在 init_dec_valid_lens 中申请，一次性构造

    int32_t *enc_valid_lens_buffer = static_cast<int32_t *>(::malloc(
        enc_valid_lens->size()
    ));
    int32_t *src_token_ids_buffer = static_cast<int32_t *>(::malloc(
        src_token_ids->size()
    ));
    int32_t *tgt_token_ids_buffer = static_cast<int32_t *>(::malloc(
        tgt_token_ids->size()
    ));
    int32_t *labels_buffer = static_cast<int32_t *>(::malloc(
        labels->size()
    ));
    float *ce_mask_buffer = static_cast<float *>(::malloc(
        ce_mask->size()
    ));

    auto res = seq2seq->forward(src_token_ids, tgt_token_ids, enc_valid_lens, dec_valid_lens);
    auto loss = res->reshape({-1, dec_vocab_size})->CrossEntropy(labels)->mask(ce_mask)->avg_1d(ce_mask);
    insert_boundary_action();
    
    std::vector<Parameter *> parameters = seq2seq->get_parameters();
    check_parameters(parameters, num_blks);
    Adam adam(parameters, lr);
    loss->backward();
    adam.clip_grad(1.0f);
    adam.step();
    // printAllActions();
    allocMemAndInitTensors();
    if (!checkpoint.empty()) {
        std::cout << "loading from checkpoint : " << checkpoint << std::endl;
        disableInitWeightAction();
        std::cout << "loaded from checkpoint" << std::endl;
    }
    std::string checkpoint_prefix = "checkpoint" + generateDateTimeSuffix();
    init_dec_valid_lens(dec_valid_lens);
    int epoch = 0;
    for (; epoch < epochs; ++epoch) {
        if (shutdown) {
            break;
        }
        float loss_sum = 0;
        int cnt = 0;
        std::string prefix = "epoch " + std::to_string(epoch) + " : ";
        for (int i = 0; i < v_src_token_ids.size(); i += batch_size) {
            cnt ++;

            auto end = i + batch_size;
            if (end > v_src_token_ids.size()) {
                break;
            }

            for (int j = i; j < end; ++j) {
                // std::cout << "j : " << j << " i : " << i << " end : " << end << std::endl;
                enc_valid_lens_buffer[j - i] = v_src_token_ids[j].size();
                auto src_j_trim_or_padding_res = trim_or_padding(
                    v_src_token_ids[j], num_steps, src_pad_id
                );
                auto tgt_j_trim_or_padding_res = trim_or_padding(
                    add_bos(v_tgt_token_ids[j], bos_id), num_steps, tgt_pad_id
                );
                auto tgt_j_labels_res = trim_or_padding(
                    v_tgt_token_ids[j], num_steps, tgt_pad_id
                );
                for (int k = 0; k < num_steps; ++k) {
                    src_token_ids_buffer[(j - i) * num_steps + k] = src_j_trim_or_padding_res[k];
                    tgt_token_ids_buffer[(j - i) * num_steps + k] = tgt_j_trim_or_padding_res[k];
                    labels_buffer[(j - i) * num_steps + k] = tgt_j_labels_res[k];
                    ce_mask_buffer[(j - i) * num_steps + k] = (tgt_j_labels_res[k] != tgt_pad_id) ? 1.0f : 0.0f;
                }
            }

            g_backend_ops->cp_to_device(
                enc_valid_lens,
                reinterpret_cast<char*>(enc_valid_lens_buffer),
                enc_valid_lens->size()
            );
            g_backend_ops->cp_to_device(
                src_token_ids,
                reinterpret_cast<char*>(src_token_ids_buffer),
                src_token_ids->size()
            );
            g_backend_ops->cp_to_device(
                tgt_token_ids,
                reinterpret_cast<char*>(tgt_token_ids_buffer),
                tgt_token_ids->size()
            );
            g_backend_ops->cp_to_device(
                labels,
                reinterpret_cast<char*>(labels_buffer),
                labels->size()
            );
            g_backend_ops->cp_to_device(
                ce_mask,
                reinterpret_cast<char*>(ce_mask_buffer),
                ce_mask->size()
            );

            gDoActions();
            print_progress(prefix, end, v_src_token_ids.size());
            float loss_v = 0;
            g_backend_ops->cp_from_device(
                reinterpret_cast<char*>(&loss_v),
                loss->get_tensor(),
                loss->get_tensor()->size()
            );
            loss_sum += loss_v;
        }
        std::cout << "loss : " << loss_sum / cnt << std::endl;
    }

    save_checkpoint(checkpoint_prefix, epoch, parameters);
    
    // free input buffers
    ::free(enc_valid_lens_buffer);
    ::free(src_token_ids_buffer);
    ::free(tgt_token_ids_buffer);
    ::free(labels_buffer);
    ::free(ce_mask_buffer);

    delete seq2seq;
    destruct_env();
    return 0;
}