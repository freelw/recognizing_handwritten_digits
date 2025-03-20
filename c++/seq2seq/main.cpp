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
#define SRC_VOCAB_TINY_NAME "../fra_vocab_builder/vocab_en_tiny.txt"
#define TGT_VOCAB_TINY_NAME "../fra_vocab_builder/vocab_fr_tiny.txt"
#define HIDDEN_SIZE 32
#define EMBED_SIZE 32
#define TINY_HIDDEN_SIZE 2
#define TINY_EMBED_SIZE 2

#define TEST_FILE "./test.txt"
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

void train(
    const std::string &corpus,
    const std::string &checkpoint,
    uint epochs,
    DATATYPE dropout,
    DATATYPE lr,
    bool tiny) {
   
    uint num_steps = 4;
    std::string src_vocab_name = tiny ? SRC_VOCAB_TINY_NAME : SRC_VOCAB_NAME;
    std::string tgt_vocab_name = tiny ? TGT_VOCAB_TINY_NAME : TGT_VOCAB_NAME;
    seq2seq::DataLoader loader(corpus, src_vocab_name, tgt_vocab_name, TEST_FILE);
    std::cout << "data loaded" << std::endl;
    uint enc_vocab_size = loader.src_vocab_size();
    // uint enc_embed_size = 32;
    // uint hidden_num = 32;
    uint enc_embed_size = tiny ? TINY_EMBED_SIZE : EMBED_SIZE;
    uint hidden_num = tiny ? TINY_HIDDEN_SIZE : HIDDEN_SIZE;
    uint layer_num = 2;
    DATATYPE sigma = 0.01;
    auto encoder = new autograd::Seq2SeqEncoder(
        enc_vocab_size, enc_embed_size, hidden_num, layer_num, sigma, dropout
    );
    uint dec_vocab_size = loader.tgt_vocab_size();
    uint dec_embed_size = enc_embed_size;
    auto decoder = new autograd::Seq2SeqDecoder(
        dec_vocab_size, dec_embed_size, hidden_num, layer_num, sigma, dropout
    );
    auto encoder_decoder = new autograd::Seq2SeqEncoderDecoder(
        encoder, decoder, loader.tgt_bos_id(), loader.tgt_eos_id()
    );
    if (!checkpoint.empty()) {
        cout << "loading from checkpoint : " << checkpoint << endl;
        loadfrom_checkpoint(*encoder_decoder, checkpoint);
        cout << "loaded from checkpoint" << endl;
    }

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
    
    auto adam = autograd::Adam(parameters, lr);
    std::string checkpoint_prefix = "checkpoint" + autograd::generateDateTimeSuffix();

    std::vector<std::vector<uint>> src_token_ids;
    std::vector<std::vector<uint>> tgt_token_ids;
    loader.get_token_ids(src_token_ids, tgt_token_ids);
    assert(src_token_ids.size() == tgt_token_ids.size());
    for (uint epoch = 0; epoch < epochs; epoch++) {
        DATATYPE loss_sum = 0;
        int emit_clip = 0;
        for (uint i = 0; i < src_token_ids.size(); i += BATCH_SIZE) {
            std::vector<std::vector<uint>> input_sentences;
            std::vector<std::vector<uint>> target_sentences;
            std::vector<std::vector<uint>> target_labels;
            std::vector<uint> labels;
            std::vector<bool> mask;
            auto end = i + BATCH_SIZE;
            if (end > src_token_ids.size()) {
                end = src_token_ids.size();
            }
            auto cur_batch_size = end - i;
            // std::cout << "prepare input" << std::endl;
            for (uint j = i; j < end; j++) {
                input_sentences.push_back(trim_or_padding(src_token_ids[j], num_steps, loader.src_pad_id()));
                target_sentences.push_back(trim_or_padding(add_bos(tgt_token_ids[j], loader.tgt_bos_id()), num_steps, loader.tgt_pad_id()));
                target_labels.push_back(trim_or_padding(tgt_token_ids[j], num_steps, loader.tgt_pad_id()));
            }

            std::vector<std::vector<uint>> inputs;
            std::vector<std::vector<uint>> targets;

            for (uint j = 0; j < num_steps; j++) {
                std::vector<uint> input;
                std::vector<uint> target;
                for (uint k = 0; k < cur_batch_size; k++) {
                    input.push_back(input_sentences[k][j]);
                    target.push_back(target_sentences[k][j]);
                }
                inputs.push_back(input);
                targets.push_back(target);
            }

            labels.reserve(cur_batch_size * num_steps);
            mask.reserve(cur_batch_size * num_steps);
            for (auto &target_label : target_labels) {
                for (auto token : target_label) {
                    labels.push_back(token);
                    mask.push_back(token != loader.tgt_pad_id());
                }
            }

            assert(inputs.size() == num_steps);
            assert(targets.size() == num_steps);
            
            // for (auto & input : inputs) {
            //     for (auto token : input) {
            //         std::cout << loader.get_src_token(token) << " ";
            //     }
            //     std::cout << std::endl;
            // }

            // for (auto & target : targets) {
            //     for (auto token : target) {
            //         std::cout << loader.get_tgt_token(token) << " ";
            //     }
            //     std::cout << std::endl;
            // }
            // std::cout << std::endl;

            // // print labels
            // for (auto & label : labels) {
            //     std::cout << loader.get_tgt_token(label) << " ";
            // }
            // std::cout << std::endl;

            // // print mask
            // for (auto m : mask) {
            //     std::cout << m << " ";
            // }
            // std::cout << std::endl;
            
            std::cout << "prepare input done" << std::endl;
            auto dec_outputs = encoder_decoder->forward(inputs, targets);
            // std::cout << "forward done" << std::endl;
            dec_outputs->checkShape(Shape(dec_vocab_size, cur_batch_size * num_steps));

            // std::cout << "dec_outputs : " << *(dec_outputs->get_weight()) << std::endl;
            // std::cout << "dec_outputs shape : " << dec_outputs->getShape() << std::endl;
            auto loss = dec_outputs->CrossEntropyMask(labels, mask);
            // auto loss = dec_outputs->CrossEntropy(labels);
            assert(loss->get_weight()->getShape().rowCnt == 1);
            assert(loss->get_weight()->getShape().colCnt == 1);
            loss_sum += (*loss->get_weight())[0][0];

            adam.zero_grad();
            loss->backward();
            if (adam.clip_grad(1)) {
                emit_clip++;
            }
            adam.step();
            // dec_outputs->cross_entropy_mask(targets, loader.tgt_pad_id());
            print_progress(end, src_token_ids.size());
            if (shutdown) {
                save_checkpoint(checkpoint_prefix, epoch, *encoder_decoder);
                exit(0);
            }
            // print all parameters grad
            // for (auto &p : parameters) {
            //     std::cout << "param : " << p->get_weight()->getShape() << std::endl;
            //     std::cout << "grad : " << *(p->get_grad()) << std::endl;
            // }
            freeTmpMatrix();
            autograd::freeAllNodes();
            autograd::freeAllEdges();
        }
        if (epoch % 10 == 0 || epoch == epochs - 1) {
            save_checkpoint(checkpoint_prefix, epoch, *encoder_decoder);
        }
        std::cout << "epoch " << epoch << " loss : " << loss_sum << " emit_clip : " << emit_clip << std::endl;
    }
    
    if (epochs > 0) {
        // pass
    } else {
        std::cout << "serving mode" << std::endl;
        encoder_decoder->train(false);
        std::vector<std::string> src_sentences = loader.get_test_sentences();
        // ./seq2seq -e 0 -c ./checkpoints/checkpoint_20250319_180713_9.bin
        for (auto & sentence : src_sentences) {
            std::vector<uint> src_token_ids = loader.to_src_token_ids(sentence);
            for (auto &token_id : src_token_ids) {
                std::cout << loader.get_src_token(token_id) << " ";
            }
            std::cout << std::endl;

            std::vector<uint> tgt_token_ids = encoder_decoder->predict(src_token_ids, 20);
            std::cout << "translate res : ";
            for (auto &token_id : tgt_token_ids) {
                std::cout << loader.get_tgt_token(token_id) << " ";
            }
            std::cout << std::endl;
        }
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
    dec_outputs->checkShape(Shape(dec_vocab_size, tgt_token_ids.size() * tgt_token_ids[0].size()));
    // assert(dec_outputs.size() == tgt_token_ids.size());
    // for (auto dec_output : dec_outputs) {
    //     assert(dec_output->getShape().rowCnt == dec_vocab_size);
    //     assert(dec_output->getShape().colCnt == batch_size);
    // }
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
    auto encoder_decoder = new autograd::Seq2SeqEncoderDecoder(encoder, decoder, 3, 1);
    auto dec_outputs = encoder_decoder->forward(src_token_ids, tgt_token_ids);
    dec_outputs->checkShape(Shape(dec_vocab_size, tgt_token_ids.size() * tgt_token_ids[0].size()));
    // assert(dec_outputs.size() == tgt_token_ids.size());
    // for (auto dec_output : dec_outputs) {
    //     assert(dec_output->getShape().rowCnt == dec_vocab_size);
    //     assert(dec_output->getShape().colCnt == src_token_ids[0].size());
    // }
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
    seq2seq::DataLoader dataloader(corpus, src_vocab, tgt_vocab, TEST_FILE);
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

void test_cat1_1() {
    Matrix *m1 = allocTmpMatrix(Shape(3, 1));
    Matrix *c1 = allocTmpMatrix(Shape(4, 1));

    m1->fill(0.1);
    c1->fill(0.2);

    (*m1)[0][0] = 0.5;
    (*c1)[0][0] = 0.5;

    autograd::Node *node1 = autograd::allocNode(m1);
    autograd::Node *node2 = autograd::allocNode(c1);

    node1->require_grad();
    node2->require_grad();

    autograd::Parameters *p1 = new autograd::Parameters(node1);
    autograd::Parameters *p2 = new autograd::Parameters(node2);

    std::vector<uint> labels_m = {0};
    std::vector<uint> labels_c = {0};

    auto loss_m = node1->CrossEntropy(labels_m);
    auto loss_c = node2->CrossEntropy(labels_c);

    std::cout << "loss_m : " << (*loss_m->get_weight())[0][0] << std::endl;
    std::cout << "loss_c : " << (*loss_c->get_weight())[0][0] << std::endl;

    autograd::Adam optimizer_m({p1}, 0.01);
    autograd::Adam optimizer_c({p2}, 0.01);

    optimizer_m.zero_grad();
    optimizer_c.zero_grad();

    loss_m->backward();
    loss_c->backward();

    optimizer_m.step();
    optimizer_c.step();

    std::cout << "node1 : " << *(node1->get_weight()) << std::endl;
    std::cout << "node1 grad : " << *(node1->get_grad()) << std::endl;
    std::cout << "node2 : " << *(node2->get_weight()) << std::endl;
    std::cout << "node2 grad : " << *(node2->get_grad()) << std::endl;

    delete p2;
    delete p1;
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_cat1_2() {

    Matrix *m1 = allocTmpMatrix(Shape(3, 1));
    Matrix *c1 = allocTmpMatrix(Shape(4, 1));

    m1->fill(0.1);
    c1->fill(0.2);

    (*m1)[0][0] = 0.5;
    (*c1)[0][0] = 0.5;

    autograd::Node *node1 = autograd::allocNode(m1);
    autograd::Node *node2 = autograd::allocNode(c1);

    node1->require_grad();
    node2->require_grad();

    autograd::Parameters *p1 = new autograd::Parameters(node1);
    autograd::Parameters *p2 = new autograd::Parameters(node2);

    std::vector<uint> labels = {0};

    auto cat_node = autograd::cat({node1, node2}, 1);

    auto loss = cat_node->CrossEntropy(labels);

    std::cout << "loss : " << (*loss->get_weight())[0][0] << std::endl;

    autograd::Adam optimizer({p1, p2}, 0.01);

    optimizer.zero_grad();

    loss->backward();

    optimizer.step();

    std::cout << "node1 : " << *(node1->get_weight()) << std::endl;
    std::cout << "node1 grad : " << *(node1->get_grad()) << std::endl;
    std::cout << "node2 : " << *(node2->get_weight()) << std::endl;
    std::cout << "node2 grad : " << *(node2->get_grad()) << std::endl;
    std::cout << "cat_node : " << *(cat_node->get_weight()) << std::endl;
    std::cout << "cat_node grad : " << *(cat_node->get_grad()) << std::endl;

    delete p2;
    delete p1;
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_cat0() {

    Matrix *m1 = allocTmpMatrix(Shape(3, 1));
    Matrix *c1 = allocTmpMatrix(Shape(3, 1));

    m1->fill(0.1);
    c1->fill(0.2);

    (*m1)[0][0] = 0.5;
    (*c1)[0][0] = 0.5;

    autograd::Node *node1 = autograd::allocNode(m1);
    autograd::Node *node2 = autograd::allocNode(c1);

    node1->require_grad();
    node2->require_grad();

    autograd::Parameters *p1 = new autograd::Parameters(node1);
    autograd::Parameters *p2 = new autograd::Parameters(node2);

    std::vector<uint> labels = {0, 0};

    auto cat_node = autograd::cat({node1, node2}, 0);

    auto loss = cat_node->CrossEntropy(labels);

    std::cout << "loss : " << (*loss->get_weight())[0][0] << std::endl;

    autograd::Adam optimizer({p1, p2}, 0.01);

    optimizer.zero_grad();

    loss->backward();

    optimizer.step();

    std::cout << "node1 : " << *(node1->get_weight()) << std::endl;
    std::cout << "node1 grad : " << *(node1->get_grad()) << std::endl;
    std::cout << "node2 : " << *(node2->get_weight()) << std::endl;
    std::cout << "node2 grad : " << *(node2->get_grad()) << std::endl;

    delete p2;
    delete p1;
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

int main(int argc, char *argv[]) {
    // test_encoder_decoder();
    // test_encoder_decoder1();
    // test_crossentropy_mask();
    // test_dataloader();
    // test_cat1_2();
    // test_cat0();
    // return 0;
    cout << "OMP_THREADS: " << OMP_THREADS << endl;
    // register signal SIGINT and signal handler
    signal(SIGINT, signal_callback_handler);

    std::string corpus;
    std::string checkpoint;
    int opt;
    uint epochs = 30;
    DATATYPE dropout = 0.2;
    DATATYPE lr = 0.005;
    bool tiny = false;
    while ((opt = getopt(argc, argv, "f:c:e:d:l:t:")) != -1) {
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
            case 'd':
                dropout = atof(optarg);
                break;
            case 'l':
                lr = atof(optarg);
                break;
            case 't':
                tiny = atoi(optarg) == 1;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f <corpus> -c <checpoint> -e <epochs>" << std::endl;
                return 1;
        }
    }

    if (corpus.empty()) {
        corpus = RESOURCE_NAME;
    }
    std::cout << "epochs : " << epochs << std::endl;
    std::cout << "dropout : " << dropout << std::endl;
    std::cout << "lr : " << lr << std::endl;
    std::cout << "tiny : " << tiny << std::endl;
    train(corpus, checkpoint, epochs, dropout, lr, tiny);
    return 0;
}