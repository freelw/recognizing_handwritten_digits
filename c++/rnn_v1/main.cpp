#include <iostream>
#include <signal.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "lmcommon/dataloader.h"
#include "rnnlm.h"
#include "getopt.h"
#include <unistd.h>

// #pragma message("warning: shutdown is true")
// bool shutdown = true; // fixme
bool shutdown = false;

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

// void save_checkpoint(const std::string & prefix, int epoch, RnnLM &lm) {
//     std::ostringstream oss;
//     oss << prefix << "_" << epoch << ".bin";
//     std::string checkpoint_name = oss.str();
//     std::string path = "./checkpoints/" + checkpoint_name;
//     auto parameters = lm.get_parameters();
//     std::ofstream out(path, std::ios::out | std::ios::binary);
//     int num_params = parameters.size();
//     out.write((char *)&num_params, sizeof(num_params));
//     for (auto p : parameters) {
//         std::string serialized = p->serialize();
//         int size = serialized.size();
//         out.write((char *)&size, sizeof(size));
//         out.write(serialized.c_str(), serialized.size());
//     }
//     out.close();
//     cout << "checkpoint saved : " << path << endl;
// }

// void loadfrom_checkpoint(RnnLM &lm, const std::string &filename) {
//     std::ifstream in(filename
//         , std::ios::in | std::ios::binary);
//     int num_params;
//     in.read((char *)&num_params, sizeof(num_params));
//     for (int i = 0; i < num_params; i++) {
//         int size;
//         in.read((char *)&size, sizeof(size));
//         char *buffer = new char[size];
//         in.read(buffer, size);
//         lm.get_parameters()[i]->deserialize(buffer);
//         delete [] buffer;
//     }
// }

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
    // std::cout << "train by " << corpus << std::endl;
    // std::cout << "epochs : " << epochs << std::endl;
    // DataLoader loader(corpus);
    // std::cout << "Data loaded" << std::endl;
    // uint num_steps = 32;
    // uint hidden_num = 32;
    // bool rand = true;
    // Rnn *rnn = new Rnn(INPUT_NUM, hidden_num, 0.01, rand);
    // RnnLM lm(rnn, INPUT_NUM, rand);
    // if (!checkpoint.empty()) {
    //     cout << "loading from checkpoint : " << checkpoint << endl;
    //     loadfrom_checkpoint(lm, checkpoint);
    //     cout << "loaded from checkpoint" << endl;
    // }
    // auto parameters = lm.get_parameters();
    // assert(parameters.size() == 5);
    // if (!rand) {
    //     init_weight(parameters[3]);
    // }
    // Adam adam(parameters, 0.001);
    // std::string checkpoint_prefix = "checkpoint" + generateDateTimeSuffix();
    // for (uint epoch = 0; epoch < epochs; epoch++) {
    //     DATATYPE loss_sum = 0;
    //     int emit_clip = 0;
    //     for (uint i = 0; i < loader.data.size() - num_steps; i++) {
    //         std::vector<Matrix *> inputs;
    //         std::vector<uint> labels;
    //         for (uint j = 0; j < num_steps; j++) {
    //             assert(i+j < loader.data.size());
    //             assert(i+j+1 < loader.labels.size());
    //             inputs.push_back(loader.data[i+j]);
    //             labels.push_back(loader.labels[i+j+1]);
    //         }
    //         assert(inputs.size() == num_steps);
    //         RnnLMContext *ctx = lm.init();
    //         Matrix *res = lm.forward(ctx, inputs);
    //         res->checkShape(Shape(INPUT_NUM, num_steps));
    //         CrossEntropyLoss loss_fn(labels);
    //         CrossEntropyLossContext *ce_ctx = (CrossEntropyLossContext *)loss_fn.init();
    //         auto loss = loss_fn.forward(ce_ctx, res);
    //         loss->checkShape(Shape(1, 1));
    //         loss_sum += (*loss)[0][0];
    //         auto grad = loss_fn.backward(ce_ctx, nullptr);
    //         loss_fn.release(ce_ctx);
    //         adam.zero_grad();
    //         lm.backward(ctx, grad);
    //         if (adam.clip_grad(1)) {
    //             emit_clip++;
    //         }
    //         adam.step();
    //         lm.release(ctx);
    //         freeTmpMatrix();
    //         if (shutdown) {
    //             save_checkpoint(checkpoint_prefix, epoch, lm);
    //             exit(0);
    //         }
    //         print_progress(i+1, loader.data.size() - num_steps);
    //     }
    //     save_checkpoint(checkpoint_prefix, epoch, lm);
    //     std::cout.precision(14);
    //     std::cout << "epoch " << epoch << " loss : " << loss_sum/(loader.data.size() - num_steps) << " emit_clip : " << emit_clip << std::endl;
    // }
    // if (epochs > 0) {
    //     // pass
    // } else {
    //     std::cout << "serving mode" << std::endl;
    // }
    // std::vector<std::string> prefixs = {
    //     "time traveller",
    //     "the time machine",
    //     "expounding a recondite",
    //     " traveller for so",
    //     "it has",
    //     "so most people",
    //     "is simply ",
    //     " we cannot move about",
    //     "and the still",
    // };
    // for (auto prefix : prefixs) {
    //     std::string predicted = lm.predict(prefix, 30);
    //     std::cout << "prefix : " << prefix << std::endl;
    //     std::cout << "predicted : " << predicted << std::endl;
    // }
    // freeTmpMatrix();
    // delete rnn;
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

void test_cat_bp() {
    Matrix *m0 = allocTmpMatrix(Shape(3, 1));
    Matrix *m1 = allocTmpMatrix(Shape(3, 1));

    m0->fill(0.1);
    m1->fill(0.1);
    (*m0)[0][0] = 0.8;
    (*m1)[0][0] = 0.8;

    std::vector<autograd::Node *> nodes;
    nodes.push_back(autograd::allocNode(m0));
    nodes.push_back(autograd::allocNode(m1));

    for (auto node : nodes) {
        node->require_grad();
    }

    auto res = autograd::cat(nodes);
    std::vector<uint> labels = {0, 1};

    auto loss = res->CrossEntropy(labels);
    std::cout << "loss : " << *loss->get_weight() << std::endl;
    loss->backward();

    for (auto node : nodes) {
        std::cout << "node : " << *node->get_grad() << std::endl;
    }


    for (auto node : nodes) {
        node->zero_grad();
    }
    auto loss0 = nodes[0]->CrossEntropy({0});
    loss0->backward();
    std::cout << "loss0 : " << *loss0->get_weight() << std::endl;
    auto loss1 = nodes[1]->CrossEntropy({1});
    std::cout << "loss1 : " << *loss1->get_weight() << std::endl;
    loss1->backward();

    auto avg = ((*loss0->get_weight())[0][0] + (*loss1->get_weight())[0][0]) / 2;
    std::cout << "avg : " << avg << std::endl;
    for (auto node : nodes) {
        std::cout << "node : " << *node->get_grad() << std::endl;
        
    }
    for (auto node : nodes) {
        std::cout << "node/2 : " << *((*node->get_grad())/2) << std::endl;
    }

    autograd::freeAllNodes();
    autograd::freeAllEdges();
    freeTmpMatrix();
}

int main(int argc, char *argv[]) {
    test_cat_bp();
    return -1;
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