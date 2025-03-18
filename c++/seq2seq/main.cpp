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

int main(int argc, char *argv[]) {
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