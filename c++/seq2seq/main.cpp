#include <iostream>
#include <fstream>
#include <signal.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "autograd/optimizers.h"
#include "seq2seq.h"
#include "getopt.h"
#include <unistd.h>
#include "stats/stats.h"

#define EMBEDDING_SIZE 32
// #pragma message("warning: shutdown is true")
// bool shutdown = true; // fixme
bool shutdown = false;

#define RESOURCE_NAME "../../resources/timemachine_preprocessed.txt"
#define VOCAB_NAME "../vocab_builder/vocab.txt"
#define BATCH_SIZE 1024

void signal_callback_handler(int signum);

std::string generateDateTimeSuffix() {    
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    struct std::tm* localTime = std::localtime(&currentTime);
    std::ostringstream oss;
    oss << std::put_time(localTime, "_%Y%m%d_%H%M%S");
    return oss.str();
}

void save_checkpoint(const std::string & prefix, int epoch, autograd::RnnLM &lm) {
    std::ostringstream oss;
    oss << prefix << "_" << epoch << ".bin";
    std::string checkpoint_name = oss.str();
    std::string path = "./checkpoints/" + checkpoint_name;
    auto parameters = lm.get_parameters();
    std::ofstream out(path, std::ios::out | std::ios::binary);
    int num_params = parameters.size();
    out.write((char *)&num_params, sizeof(num_params));
    for (auto p : parameters) {
        std::string serialized = p->serialize();
        int size = serialized.size();
        out.write((char *)&size, sizeof(size));
        out.write(serialized.c_str(), serialized.size());
    }
    out.close();
    cout << "checkpoint saved : " << path << endl;
}

void loadfrom_checkpoint(autograd::RnnLM &lm, const std::string &filename) {
    std::ifstream in(filename
        , std::ios::in | std::ios::binary);
    // check file exsit
    if (!in) {
        std::cerr << "file not found : " << filename << std::endl;
        exit(1);
    }
    int num_params = 0;    
    in.read((char *)&num_params, sizeof(num_params));
    for (int i = 0; i < num_params; i++) {
        int size;
        in.read((char *)&size, sizeof(size));
        char *buffer = new char[size];
        in.read(buffer, size);
        lm.get_parameters()[i]->deserialize(buffer);
        delete [] buffer;
    }
}

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