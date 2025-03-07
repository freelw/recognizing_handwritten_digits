#include <iostream>

#include "dataloader.h"
#include "rnnlm.h"
#include "optimizers/optimizers.h"

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

int main(int argc, char *argv[]) {
    bool test = false;
    bool testdl = false;
    bool testce = false;
    if (argc == 2) {
        if (std::string(argv[1]) == "test") {
            test = true;
        } else if (std::string(argv[1]) == "testdl") {
            testdl = true;
        } else if (std::string(argv[1]) == "testce") {
            testce = true;
        }
    }

    if (test) {
        testgrad();
    } else if (testdl) {
        load_data();
    } else if (testce) {
        testcrossentropy();
    } else {
        DataLoader loader(RESOURCE_NAME);
        std::cout << "Data loaded" << std::endl;
        uint num_steps = 32;
        uint hidden_num = 32;
        bool rand = true;
        Rnn *rnn = new Rnn(INPUT_NUM, hidden_num, 0.01, rand);
        RnnLM lm(rnn, INPUT_NUM, rand);
        auto parameters = lm.get_parameters();
        if (!rand) {
            init_weight(parameters[3]);
        }
        
        Adam adam(parameters, 0.001);
        for (uint epoch = 0; epoch < 30; epoch++) {
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
            }
            std::cout << "epoch " << epoch << " loss : " << loss_sum/(loader.data.size() - num_steps) << std::endl;
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
    return 0;
}