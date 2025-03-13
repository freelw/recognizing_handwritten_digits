#include <iostream>
#include <signal.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "dataloader.h"
#include "autograd/optimizers.h"
#include "rnnlm.h"
#include "getopt.h"
#include <unistd.h>

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
    int num_params;
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

void test_print_progress() {
    for (uint i = 0; i < 100; i++) {
        print_progress(i+1, 100);
        sleep(1);
        if (shutdown) {
            break;
        }
    }
}

void gen_matrix_labels(const std::string &content, uint offset, uint len, Matrix * &input, std::vector<uint> &labels) {
    assert(offset + len < content.size());
    input = allocTmpMatrix(Shape(INPUT_NUM, len));
    for (uint i = 0; i < len; i++) {
        uint pos = offset + i;
        char ch = content[pos];
        char next_ch = content[pos+1];
        assert((ch >= 'a' && ch <= 'z') || ch == ' ');
        (*input)[to_index(ch)][i] = 1;
        labels.push_back(to_index(next_ch));
    }
}

int gen_batch(
    int start,
    gru::DataLoader &loader,
    uint num_steps,
    uint batch_size,
    std::vector<autograd::Node *> &inputs,
    std::vector<uint> &whole_labels) {
    /*
        这里第二个参数不能+1，因为还有留一个label
        比如 content 长度为2
        start = 0, num_steps = 2
        这样即便用batch=1，最后一个元素也获取不到label
    */
    int cur_batch_size = std::min(batch_size, (int)loader.size() - start - num_steps); 
    
    if (cur_batch_size <= 0) {
        return 0;
    }
    for (uint i = 0; i < num_steps; i++) {
        Matrix *input = nullptr;
        std::vector<uint> labels;
        gen_matrix_labels(loader.content, start+i, cur_batch_size, input, labels);
        assert(input->getShape().colCnt == (uint)cur_batch_size);
        inputs.push_back(autograd::allocNode(input));
        whole_labels.insert(whole_labels.end(), labels.begin(), labels.end());
    }
    return cur_batch_size;
}

void train(const std::string &corpus, const std::string &checkpoint, uint epochs) {
    std::cout << "train by " << corpus << std::endl;
    std::cout << "epochs : " << epochs << std::endl;
    gru::DataLoader loader(corpus, VOCAB_NAME);
    std::cout << "Data loaded" << std::endl;
    uint num_steps = 32;
    uint hidden_num = 32;
    autograd::GRU *rnn = new autograd::GRU(EMBEDDING_SIZE, hidden_num, 0.01);
    autograd::Embedding *embedding = new autograd::Embedding(loader.vocab_size(), EMBEDDING_SIZE);
    autograd::RnnLM lm(rnn, embedding, EMBEDDING_SIZE);
    if (!checkpoint.empty()) {
        cout << "loading from checkpoint : " << checkpoint << endl;
        loadfrom_checkpoint(lm, checkpoint);
        cout << "loaded from checkpoint" << endl;
    }
    auto parameters = lm.get_parameters();
    assert(parameters.size() == 12);
    
    autograd::Adam adam(parameters, 0.001);
    std::string checkpoint_prefix = "checkpoint" + generateDateTimeSuffix();
    for (uint epoch = 0; epoch < epochs; epoch++) {
        DATATYPE loss_sum = 0;
        int emit_clip = 0;
        int i = 0;
        int loops = 0;
        while (1) {
            std::vector<autograd::Node *> inputs;
            std::vector<uint> whole_labels;
            int ret = gen_batch(i, loader, num_steps, BATCH_SIZE, inputs, whole_labels);
            if (ret == 0){
                break;
            }
            i += ret;
            loops++;
            auto loss = lm.forward(inputs)->CrossEntropy(whole_labels);
            assert(loss->getShape().rowCnt == 1);
            assert(loss->getShape().colCnt == 1);
            loss_sum += (*loss->get_weight())[0][0];
            adam.zero_grad();
            loss->backward();
            if (adam.clip_grad(1)) {
                emit_clip++;
            }
            adam.step();
            autograd::freeAllNodes();
            autograd::freeAllEdges();
            freeTmpMatrix();
            if (shutdown) {
                save_checkpoint(checkpoint_prefix, epoch, lm);
                exit(0);
            }
            print_progress(i, loader.content.length() - num_steps);
        }
        save_checkpoint(checkpoint_prefix, epoch, lm);
        std::cout.precision(14);
        std::cout << "epoch " << epoch << " loss : " << loss_sum/loops << " emit_clip : " << emit_clip << std::endl;
    }
    if (epochs > 0) {
        // pass
    } else {
        std::cout << "serving mode" << std::endl;
    }
    std::vector<std::string> prefixs = {
        "time traveller",
        "the time machine",
        "expounding a recondite",
        " traveller for so",
        "it has",
        "so most people",
        "is simply ",
        " we cannot move about",
        "and the still",
    };
    for (auto prefix : prefixs) {
        std::string predicted = lm.predict(prefix, 30);
        std::cout << "prefix : " << prefix << std::endl;
        std::cout << "predicted : " << predicted << std::endl;
    }
    delete embedding;
    delete rnn;
    autograd::freeAllNodes();
    autograd::freeAllEdges();
    freeTmpMatrix();
}

void test_cat() {
    std::vector<autograd::Node *> nodes;
    for (uint i = 0; i < 2; i++) {
        Matrix *m = allocTmpMatrix(Shape(3, 10));
        for (uint j = 0; j < 3; j++) {
            for (uint k = 0; k < 10; k++) {
                (*m)[j][k] = i*100+j*10+k;
            }
        }
        nodes.push_back(autograd::allocNode(m));
    }
    for (auto node : nodes) {
        std::cout << "node : " << *node->get_weight() << std::endl;
    }

    auto res = autograd::cat(nodes);
    std::cout << "res : " << *res->get_weight() << std::endl;
    
    autograd::freeAllNodes();
    autograd::freeAllEdges();
    
    // assert(res)
    freeTmpMatrix();
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

    if (corpus.empty()) {
        corpus = RESOURCE_NAME;
    }

    train(corpus, checkpoint, epochs);
    return 0;
}