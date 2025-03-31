#include <iostream>
#include <signal.h>
#include <unistd.h>
#include "layernorm.h"
#include "attention.h"
#include "posencoding.h"
#include "addnorm.h"
#include "ffn.h"
#include "test.h"
#include "dataloader.h"
#include "optimizers.h"

using namespace std;

bool shutdown = false;

#define RESOURCE_NAME "../../resources/fra_preprocessed.txt"
#define SRC_VOCAB_NAME "../fra_vocab_builder/vocab_en.txt"
#define TGT_VOCAB_NAME "../fra_vocab_builder/vocab_fr.txt"
#define SRC_VOCAB_TINY_NAME "../fra_vocab_builder/vocab_en_tiny.txt"
#define TGT_VOCAB_TINY_NAME "../fra_vocab_builder/vocab_fr_tiny.txt"
#define HIDDEN_SIZE 256
#define EMBED_SIZE 256
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
    
}

int main(int argc, char *argv[]) {
    cout << "OMP_THREADS: " << OMP_THREADS << endl;
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