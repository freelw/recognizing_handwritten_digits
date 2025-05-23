#include "layers/layers.h"
#include "optimizers/optimizers.h"
#include "layers/rnnlm.h"
#include <iostream>


void init_weight(Parameters *p) {
    std::cerr << "[warning] init weight" << std::endl;
    int cnt = 1;
    auto w = p->get_weight();
    for (uint i = 0; i < w->getShape().rowCnt; ++ i) {
        for (uint j = 0; j < w->getShape().colCnt; ++ j) {
            (*w)[i][j] = 0.1*cnt;
            cnt ++;
        }
    }
}

void testgrad() {
    uint vocab_size = 3;
    std::vector<Matrix *> inputs;
    for (int i = 0; i < 4; ++ i) {
        inputs.push_back(new Matrix(Shape(vocab_size, 1)));
    }

    (*(inputs[0]))[0][0] = 1;
    (*(inputs[1]))[1][0] = 1;
    (*(inputs[2]))[2][0] = 1;
    (*(inputs[3]))[0][0] = 1;

    Rnn *rnn = new Rnn(vocab_size, 4, 0.01, false);
    RnnLM lm(rnn, 3, false);
    
    auto parameters = lm.get_parameters();
    init_weight(parameters[3]);
    cout << "parameters[3] : " << *(parameters[3]->get_weight()) << endl;
    Adam adam(parameters, 0.001);
    for (auto e = 0; e < 30; ++ e) {

        RnnLMContext *ctx = lm.init();
        Matrix *res = lm.forward(ctx, inputs);
        // std::cout << "res : " << *res << std::endl;
        CrossEntropyLoss loss_fn({2, 1, 2, 0});
        // CrossEntropyLoss loss_fn({2, 1});
        // CrossEntropyLoss loss_fn({2});
        CrossEntropyLossContext *ce_ctx = (CrossEntropyLossContext *)loss_fn.init();
        auto loss = loss_fn.forward(ce_ctx, res);
        std::cout << "e : " << e << " loss : " << *loss;
        auto grad = loss_fn.backward(ce_ctx, nullptr);
        loss_fn.release(ce_ctx);
        // std::cout << "grad : " << *grad << std::endl;
        adam.zero_grad();
        lm.backward(ctx, grad);
        adam.clip_grad(1);
        adam.step();
        lm.release(ctx);
    }
    // print all parameters
    // for (auto p : parameters) {
    //     std::cout << *p << std::endl;
    // }
    
    for (int i = 0; i < 4; ++ i) {
        delete inputs[i];
    }
    delete rnn;
    freeTmpMatrix();
}