#include <iostream>

#include "dataloader.h"
#include "rnnlm.h"
#include "optimizers/optimizers.h"

void testgrad();

#define RESOURCE_NAME "../../resources/timemachine_preprocessed.txt"

void load_data() {
    DataLoader loader(RESOURCE_NAME);
}

int main(int argc, char *argv[]) {
    bool test = false;
    bool testdl = false;
    if (argc == 2) {
        if (std::string(argv[1]) == "test") {
            test = true;
        } else if (std::string(argv[1]) == "testdl") {
            testdl = true;
        }
    }

    if (test) {
        testgrad();
    } else if (testdl) {
        load_data();
    } else {
        DataLoader loader(RESOURCE_NAME);

        std::cout << "Data loaded" << std::endl;
        uint num_steps = 32;
        uint hidden_num = 32;

        Rnn *rnn = new Rnn(INPUT_NUM, hidden_num, 0.01);
        RnnLM lm(rnn, INPUT_NUM);
        RnnLMContext *ctx = lm.init();
        Adam adam(lm.get_parameters(), 0.001);
        DATATYPE loss_sum = 0;
        for (uint epoch = 0; epoch < 100; epoch++) {
            for (uint i = 0; i < loader.data.size() - num_steps; i++) {
                std::vector<Matrix *> inputs;
                for (uint j = 0; j < num_steps; j++) {
                    inputs.push_back(loader.data[i+j]);
                }
                std::vector<uint> labels;
                for (uint j = 0; j < num_steps; j++) {
                    labels.push_back(loader.labels[i+j+1]);
                }
                
                Matrix *res = lm.forward(ctx, inputs);
                CrossEntropyLoss loss_fn(labels);
                CrossEntropyLossContext *ce_ctx = (CrossEntropyLossContext *)loss_fn.init();
                auto loss = loss_fn.forward(ce_ctx, res);
                loss_sum += (*loss)[0][0];
                auto grad = loss_fn.backward(ce_ctx, nullptr);
                loss_fn.release(ce_ctx);
                lm.zero_grad();
                lm.backward(ctx, grad);
                lm.clip_grad(1);
                adam.step();
            }
            std::cout << "epoch " << epoch << " loss : " << loss_sum/loader.data.size() << std::endl;   
        }
        lm.release(ctx);
    }
    return 0;
}