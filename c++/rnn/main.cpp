#include <iostream>
#include <signal.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "dataloader.h"
#include "rnnlm.h"
#include "optimizers/optimizers.h"
#include "getopt.h"
#include <unistd.h>

// #pragma message("warning: shutdown is true")
// bool shutdown = true; // fixme
bool shutdown = false;

void testgrad();
void testcrossentropy();
void init_weight(Parameters *p);

#define RESOURCE_NAME "../../resources/timemachine_preprocessed.txt"

void load_data() {
    DataLoader loader(RESOURCE_NAME);
}

void print_input(const std::vector<Matrix *> &inputs,
                std::vector<uint> &labels, std::string &content) {
    for (uint i = 0; i < inputs.size(); i++) {
        int hot_cnt = 0;
        assert(inputs[i]->getShape().rowCnt == INPUT_NUM);
        assert(inputs[i]->getShape().colCnt == 1);
        for (uint j = 0; j < inputs[i]->getShape().rowCnt; j++) {
            if ((*inputs[i])[j][0] == 1) {
                hot_cnt++;
                std::cout << "hot : " << j << " ch : " << content[i] << " label : " << labels[i] << std::endl;
            }
        }
        assert(hot_cnt == 1);
    }
}

void signal_callback_handler(int signum);


std::string generateDateTimeSuffix() {    
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    struct std::tm* localTime = std::localtime(&currentTime);
    std::ostringstream oss;
    oss << std::put_time(localTime, "_%Y%m%d_%H%M%S");
    return oss.str();
}

void save_checkpoint(RnnLM &lm) {
    std::cout << "shutting down" << std::endl;
    std::string checkpoint_name = "checkpoint" + generateDateTimeSuffix() + ".bin";
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

void loadfrom_checkpoint(RnnLM &lm, std::string filename) {
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

void train(const std::string &corpus, const std::string &checkpoint, uint epochs) {
    std::cout << "train by " << corpus << std::endl;
    std::cout << "epochs : " << epochs << std::endl;
    DataLoader loader(corpus);
    std::cout << "Data loaded" << std::endl;
    uint num_steps = 32;
    uint hidden_num = 32;
    bool rand = true;
    Rnn *rnn = new Rnn(INPUT_NUM, hidden_num, 0.01, rand);
    RnnLM lm(rnn, INPUT_NUM, rand);
    if (!checkpoint.empty()) {
        cout << "loading from checkpoint : " << checkpoint << endl;
        loadfrom_checkpoint(lm, checkpoint);
        cout << "loaded from checkpoint" << endl;
    }
    auto parameters = lm.get_parameters();
    if (!rand) {
        init_weight(parameters[3]);
    }
    Adam adam(parameters, 0.001);
    for (uint epoch = 0; epoch < epochs; epoch++) {
        DATATYPE loss_sum = 0;
        for (uint i = 0; i < loader.data.size() - num_steps; i++) {
            std::vector<Matrix *> inputs;
            std::vector<uint> labels;
            for (uint j = 0; j < num_steps; j++) {
                assert(i+j < loader.data.size());
                assert(i+j+1 < loader.labels.size());
                inputs.push_back(loader.data[i+j]);
                labels.push_back(loader.labels[i+j+1]);
            }
            assert(inputs.size() == num_steps);
            RnnLMContext *ctx = lm.init();
            Matrix *res = lm.forward(ctx, inputs);
            res->checkShape(Shape(INPUT_NUM, num_steps));
            CrossEntropyLoss loss_fn(labels);
            CrossEntropyLossContext *ce_ctx = (CrossEntropyLossContext *)loss_fn.init();
            auto loss = loss_fn.forward(ce_ctx, res);
            loss->checkShape(Shape(1, 1));
            loss_sum += (*loss)[0][0];
            auto grad = loss_fn.backward(ce_ctx, nullptr);
            loss_fn.release(ce_ctx);
            lm.zero_grad();
            lm.backward(ctx, grad);
            lm.clip_grad(1);
            adam.step();
            lm.release(ctx);
            freeTmpMatrix();
            if (shutdown) {
                save_checkpoint(lm);
                exit(0);
            }
            print_progress(i+1, loader.data.size() - num_steps);
        }
        std::cout << "epoch " << epoch << " loss : " << loss_sum/(loader.data.size() - num_steps) << std::endl;
    }
    if (epochs > 0) {
        save_checkpoint(lm);
    } else {
        std::cout << "serving mode" << std::endl;
    }
    std::vector<std::string> prefixs = {
        "time traveller",
        "the time machine",
    };
    for (auto prefix : prefixs) {
        std::string predicted = lm.predict(prefix, 10);
        std::cout << "prefix : " << prefix << std::endl;
        std::cout << "predicted : " << predicted << std::endl;
    }
    freeTmpMatrix();
    delete rnn;
}

int main(int argc, char *argv[]) {
    // register signal SIGINT and signal handler
    signal(SIGINT, signal_callback_handler);

    std::string corpus;
    std::string checkpoint;
    std::string testcase;
    int opt;
    uint epochs = 30;
    while ((opt = getopt(argc, argv, "f:c:t:e:")) != -1) {
        switch (opt) {
            case 'f':
                corpus = optarg;
                break;
            case 'c':
                checkpoint = optarg;
                break;
            case 't':
                testcase = optarg;
                break;
            case 'e':
                std::cout << "epochs : " << optarg << std::endl;
                epochs = atoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f <corpus> -c <checpoint> -t <testcase> -e <epochs>" << std::endl;
                return 1;
        }
    }

    if (!testcase.empty()) {
        if (testcase == "test") {
            testgrad();
            return 0;
        } else if (testcase == "testdl") {
            load_data();
            return 0;
        } else if (testcase == "testce") {
            testcrossentropy();
            return 0;
        } else if (testcase == "testprog") {
            test_print_progress();
            return 0;
        } else {
            std::cerr << "not supported testcase" << std::endl;
            return 1;
        }
    }

    if (corpus.empty()) {
        corpus = RESOURCE_NAME;
    }

    train(corpus, checkpoint, epochs);
    return 0;
}