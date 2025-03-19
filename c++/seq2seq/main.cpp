#include <iostream>

#include <signal.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "autograd/optimizers.h"
#include "autograd/node.h"
#include "seq2seq.h"
#include "checkpoint.h"
#include "getopt.h"
#include <unistd.h>
#include "stats/stats.h"
#include "dataloader.h"
#include "checkpoint.h"
#include <algorithm>

bool shutdown = false;

#define RESOURCE_NAME "../../resources/fra_preprocessed.txt"
#define SRC_VOCAB_NAME "../fra_vocab_builder/vocab_en.txt"
#define TGT_VOCAB_NAME "../fra_vocab_builder/vocab_fr.txt"
#define BATCH_SIZE 128

void signal_callback_handler(int signum);

void print_progress(uint i, uint tot) {
    std::cout << "\r[" << i << "/" << tot << "]" << std::flush;
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

void train(const std::string &corpus, const std::string &checkpoint, uint epochs) {
   
    uint num_steps = 9;
    seq2seq::DataLoader loader(corpus, SRC_VOCAB_NAME, TGT_VOCAB_NAME);
    uint enc_vocab_size = loader.src_vocab_size();
    uint enc_embed_size = 256;
    uint hidden_num = 256;
    uint layer_num = 2;
    DATATYPE sigma = 0.01;
    DATATYPE dropout = 0.2;
    auto encoder = new autograd::Seq2SeqEncoder(
        enc_vocab_size, enc_embed_size, hidden_num, layer_num, sigma, dropout
    );
    uint dec_vocab_size = loader.tgt_vocab_size();
    uint dec_embed_size = enc_embed_size;
    auto decoder = new autograd::Seq2SeqDecoder(
        dec_vocab_size, dec_embed_size, hidden_num, layer_num, sigma, dropout
    );
    auto encoder_decoder = new autograd::Seq2SeqEncoderDecoder(encoder, decoder);

    auto parameters = encoder_decoder->get_parameters();

    assert(parameters.size() == 
        enc_vocab_size + dec_vocab_size // vocab embeddings
            + layer_num * (
                3 // encoder wxh + whh + b
                + 3 // encoder wxr + whr + b
                + 3 // encoder wxz + whz + b
                + 3 // decoder wxh + whh + b
                + 3 // decoder wxr + whr + b
                + 3 // decoder wxz + whz + b
            ) + 2 // output wx + b
        );
    
    auto adam = autograd::Adam(parameters, 0.005);
    std::string checkpoint_prefix = "checkpoint" + autograd::generateDateTimeSuffix();

    std::vector<std::vector<uint>> src_token_ids;
    std::vector<std::vector<uint>> tgt_token_ids;
    loader.get_token_ids(src_token_ids, tgt_token_ids);
    assert(src_token_ids.size() == tgt_token_ids.size());
    for (uint epoch = 0; epoch < epochs; epoch++) {
        DATATYPE loss_sum = 0;
        int emit_clip = 0;
        for (uint i = 0; i < src_token_ids.size(); i += BATCH_SIZE) {
            std::vector<std::vector<uint>> inputs;
            std::vector<std::vector<uint>> targets;
            auto end = i + BATCH_SIZE;
            if (end > src_token_ids.size()) {
                end = src_token_ids.size();
            }
            for (uint j = i; j < end; j++) {
                inputs.push_back(trim_or_padding(src_token_ids[j], num_steps, loader.src_pad_id()));
                targets.push_back(trim_or_padding(add_bos(tgt_token_ids[j], loader.tgt_bos_id()), num_steps, loader.tgt_pad_id()));   
            }

            auto dec_outputs = encoder_decoder->forward(inputs, targets);
            // dec_outputs->cross_entropy_mask(targets, loader.tgt_pad_id());
            print_progress(end, src_token_ids.size());
        }
        autograd::save_checkpoint(checkpoint_prefix, epoch, *encoder_decoder);
        if (shutdown) {
            break;
        }
        std::cout << "epoch " << epoch << " loss : " << loss_sum << " emit_clip : " << emit_clip << std::endl;
    }
    delete encoder_decoder;
    delete decoder;
    delete encoder;
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_encoder_decoder() {
    uint enc_vocab_size = 20;
    uint enc_embed_size = 8;
    uint hidden_num = 16;
    uint layer_num = 2;
    DATATYPE sigma = 0.01;
    DATATYPE dropout = 0.2;
    auto encoder = new autograd::Seq2SeqEncoder(
        enc_vocab_size, enc_embed_size, hidden_num, layer_num, sigma, dropout
    );
    std::vector<std::vector<uint>> src_token_ids;
    src_token_ids.push_back({0, 1, 2, 3}); // step 1
    src_token_ids.push_back({4, 5, 6, 7}); // step 2
    src_token_ids.push_back({8, 9, 10, 11}); // step 3
    for (uint i = 0; i < src_token_ids.size(); i++) {
        for (uint j = 0; j < src_token_ids[i].size(); j++) {
            assert(src_token_ids[i][j] < enc_vocab_size);
        }
    }
    std::vector<autograd::Node *> encoder_states;
    auto hiddens = encoder->forward(src_token_ids, encoder_states);
    assert(encoder_states.size() == layer_num);
    assert(encoder_states[0]->getShape().rowCnt == hidden_num);
    assert(encoder_states[0]->getShape().colCnt == src_token_ids[0].size());
    assert(hiddens.size() == src_token_ids.size());
    assert(src_token_ids[0].size() > 0);
    uint batch_size = src_token_ids[0].size();
    for (auto hidden : hiddens) {
        assert(hidden->get_weight()->getShape().rowCnt == hidden_num);
        assert(hidden->get_weight()->getShape().colCnt == batch_size);
        // std::cout << "hidden : " << hidden->getShape() << std::endl;
    }
    uint dec_vocab_size = 28;
    uint dec_embed_size = 8;
    auto decoder = new autograd::Seq2SeqDecoder(
        dec_vocab_size, dec_embed_size, hidden_num, layer_num, sigma, dropout
    );
    std::vector<std::vector<uint>> tgt_token_ids;
    tgt_token_ids.push_back({12, 13, 14, 15}); // step 1
    tgt_token_ids.push_back({17, 18, 19, 20}); // step 2
    tgt_token_ids.push_back({22, 23, 24, 25}); // step 3
    auto ctx = hiddens.back();
    assert(ctx->getShape().rowCnt == hidden_num);
    assert(ctx->getShape().colCnt == batch_size);
    auto dec_outputs = decoder->forward(tgt_token_ids, ctx, encoder_states);
    assert(dec_outputs.size() == tgt_token_ids.size());
    for (auto dec_output : dec_outputs) {
        assert(dec_output->getShape().rowCnt == dec_vocab_size);
        assert(dec_output->getShape().colCnt == batch_size);
    }
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
    delete decoder;
    delete encoder;
}

void test_encoder_decoder1() {
    std::vector<std::vector<uint>> src_token_ids;
    src_token_ids.push_back({0, 1, 2, 3}); // step 1
    src_token_ids.push_back({4, 5, 6, 7}); // step 2
    src_token_ids.push_back({8, 9, 10, 11}); // step 3
    std::vector<std::vector<uint>> tgt_token_ids;
    tgt_token_ids.push_back({12, 13, 14, 15}); // step 1
    tgt_token_ids.push_back({17, 18, 19, 20}); // step 2
    tgt_token_ids.push_back({22, 23, 24, 25}); // step 3
    uint enc_vocab_size = 20;
    uint enc_embed_size = 8;
    uint hidden_num = 16;
    uint layer_num = 2;
    DATATYPE sigma = 0.01;
    DATATYPE dropout = 0.2;
    auto encoder = new autograd::Seq2SeqEncoder(
        enc_vocab_size, enc_embed_size, hidden_num, layer_num, sigma, dropout
    );
    uint dec_vocab_size = 28;
    uint dec_embed_size = 8;
    auto decoder = new autograd::Seq2SeqDecoder(
        dec_vocab_size, dec_embed_size, hidden_num, layer_num, sigma, dropout
    );
    auto encoder_decoder = new autograd::Seq2SeqEncoderDecoder(encoder, decoder);
    auto dec_outputs = encoder_decoder->forward(src_token_ids, tgt_token_ids);
    assert(dec_outputs.size() == tgt_token_ids.size());
    for (auto dec_output : dec_outputs) {
        assert(dec_output->getShape().rowCnt == dec_vocab_size);
        assert(dec_output->getShape().colCnt == src_token_ids[0].size());
    }
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
    delete encoder_decoder;
    delete decoder;
    delete encoder;
}

void test_crossentropy_mask() {

    Matrix *input = allocTmpMatrix(Shape(3, 4));
    input->fill(0.1);
    (*input)[0][0] = 0.2;
    std::vector<uint> labels = {0, 1, 2, 2};
    Matrix *input1 = allocTmpMatrix(Shape(3, 5));
    input1->fill(0.1);
    (*input1)[0][0] = 0.2;
    std::vector<uint> labels1 = {0, 1, 2, 2, 2};

    Matrix *inputMask = allocTmpMatrix(Shape(3, 5));
    inputMask->fill(0.1);
    (*inputMask)[0][0] = 0.2;
    std::vector<uint> labelsMask = {0, 1, 2, 2, 2};
    
    std::vector<bool> mask = {true, true, true, true, false};

    autograd::Node *node = autograd::allocNode(input);
    node->require_grad();
    autograd::Parameters *p = new autograd::Parameters(node);
    
    autograd::Node *node1 = autograd::allocNode(input1);
    node1->require_grad();
    autograd::Parameters *p1 = new autograd::Parameters(node1);

    autograd::Node *node_mask = autograd::allocNode(inputMask);
    node_mask->require_grad();
    autograd::Parameters *p_mask = new autograd::Parameters(node_mask);
    
    auto loss = node->CrossEntropy(labels);
    auto loss1 = node1->CrossEntropy(labels1);
    auto loss_mask = node_mask->CrossEntropyMask(labelsMask, mask);

    std::cout << "loss : " << (*loss->get_weight())[0][0] << std::endl;
    std::cout << "loss1 : " << (*loss1->get_weight())[0][0] << std::endl;
    std::cout << "loss_mask : " << (*loss_mask->get_weight())[0][0] << std::endl;


    autograd::Adam optimizer({p}, 0.01);
    autograd::Adam optimizer1({p1}, 0.01);
    autograd::Adam optimizer_mask({p_mask}, 0.01);

    optimizer.zero_grad();
    optimizer1.zero_grad();
    optimizer_mask.zero_grad();

    loss->backward();
    loss1->backward();
    loss_mask->backward();

    optimizer.step();
    optimizer1.step();
    optimizer_mask.step();

    std::cout << "node : " << *(node->get_weight()) << std::endl;
    std::cout << "node grad : " << *(node->get_grad()) << std::endl;

    std::cout << "node1 : " << *(node1->get_weight()) << std::endl;
    std::cout << "node1 grad : " << *(node1->get_grad()) << std::endl;

    std::cout << "node_mask : " << *(node_mask->get_weight()) << std::endl;
    std::cout << "node_mask grad : " << *(node_mask->get_grad()) << std::endl;

    delete p_mask;
    delete p1;
    delete p;
    
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_dataloader() {
    std::string corpus = RESOURCE_NAME;
    std::string src_vocab = SRC_VOCAB_NAME;
    std::string tgt_vocab = TGT_VOCAB_NAME;
    seq2seq::DataLoader dataloader(corpus, src_vocab, tgt_vocab);
    std::vector<std::vector<uint>> src_token_ids;
    std::vector<std::vector<uint>> tgt_token_ids;
    dataloader.get_token_ids(src_token_ids, tgt_token_ids);
    
    std::cout << "src_token_ids.size() : " << src_token_ids.size() << std::endl;
    std::cout << "tgt_token_ids.size() : " << tgt_token_ids.size() << std::endl;

    std::cout << "src back : " << std::endl;
    for (auto &token : src_token_ids.back()) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    for (auto &token : src_token_ids.back()) {
        std::cout << dataloader.get_src_token(token) << " ";
    }
    std::cout << std::endl;

    std::cout<< "tgt back : " << std::endl;
    for (auto &token : tgt_token_ids.back()) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    for (auto &token : tgt_token_ids.back()) {
        std::cout << dataloader.get_tgt_token(token) << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    // test_encoder_decoder();
    // test_encoder_decoder1();
    // test_crossentropy_mask();
    // test_dataloader();
    // return 0;
    cout << "OMP_THREADS: " << OMP_THREADS << endl;
    // register signal SIGINT and signal handler
    signal(SIGINT, signal_callback_handler);

    std::string corpus;
    std::string checkpoint;
    int opt;
    uint epochs = 30;
    while ((opt = getopt(argc, argv, "f:c:e:")) != -1) {
        switch (opt) {
            case 'f':
                corpus = optarg;
                break;
            case 'c':
                checkpoint = optarg;
                break;
            case 'e':
                std::cout << "epochs : " << optarg << std::endl;
                epochs = atoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f <corpus> -c <checpoint> -e <epochs>" << std::endl;
                return 1;
        }
    }

    if (corpus.empty()) {
        corpus = RESOURCE_NAME;
    }

    train(corpus, checkpoint, epochs);

    return 0;
}