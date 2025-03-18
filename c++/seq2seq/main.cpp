#include <iostream>

#include <signal.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "autograd/optimizers.h"
#include "seq2seq.h"
#include "checkpoint.h"
#include "getopt.h"
#include <unistd.h>
#include "stats/stats.h"

#define EMBEDDING_SIZE 32
// #pragma message("warning: shutdown is true")
// bool shutdown = true; // fixme
bool shutdown = false;

#define RESOURCE_NAME "../../resources/fra_preprocessed.txt"
#define SRC_VOCAB_NAME "../fra_vocab_builder/fra_src_vocab.txt"
#define TGT_VOCAB_NAME "../fra_vocab_builder/fra_tgt_vocab.txt"
#define BATCH_SIZE 128

void signal_callback_handler(int signum);

void print_progress(uint i, uint tot) {
    std::cout << "\r[" << i << "/" << tot << "]" << std::flush;
}

void train(const std::string &corpus, const std::string &checkpoint, uint epochs) {
   
}

void test_encoder() {
// vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
// batch_size, num_steps = 4, 9
// encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
// X = torch.zeros((batch_size, num_steps))
// enc_outputs, enc_state = encoder(X)
// d2l.check_shape(enc_outputs, (num_steps, batch_size, num_hiddens))
    uint vocab_size = 10;
    uint embed_size = 8;
    uint hidden_num = 16;
    uint layer_num = 2;
    DATATYPE sigma = 0.01;
    DATATYPE dropout = 0.2;
    auto encoder = new autograd::Seq2SeqEncoder(
        vocab_size, embed_size, hidden_num, layer_num, sigma, dropout
    );
    std::vector<std::vector<uint>> token_ids;
    token_ids.push_back({0, 1, 2, 3});
    token_ids.push_back({4, 5, 6, 7});
    auto res = encoder->forward(token_ids);
    auto size = res.size();
    assert(size == layer_num);
    auto hiddens = res[size - 1];
    assert(hiddens.size() == token_ids.size());
    for (auto hidden : hiddens) {
        std::cout << "hidden : " << hidden->getShape() << std::endl;
    }
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
    delete encoder;
}

int main(int argc, char *argv[]) {
    test_encoder();
    return 0;
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
    return 0;
}