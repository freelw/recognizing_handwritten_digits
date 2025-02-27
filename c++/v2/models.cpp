#include "models.h"

MLP::MLP(uint _input, const std::vector<uint> &_outputs) 
    : input(_input), outputs(_outputs)
{

}

void MLP::init() {
    uint x = input;
    for (uint i = 0; i < outputs.size(); ++ i) {
        uint y = outputs[i];
        layers.push_back(new Liner(x, y));
        x = y;
        if (i < outputs.size()-1) {
            layers.push_back(new Relu());
        }
    }

    for (auto &layer : layers) {
        ctxs.push_back(layer->init());
    }
}

void MLP::destroy() {
    for (uint i = 0; i < layers.size(); ++ i) {
        auto &ctx = ctxs[i];
        layers[i]->release(ctx);
        delete layers[i];
    }
}

Matrix *MLP::forward(Matrix *input) {
    Matrix *res = allocTmpMatrix(input);
    for (uint i = 0; i < layers.size(); ++ i) {
        res = layers[i]->forward(ctxs[i], res);
    }
    return res;
}

void MLP::backward(Matrix *input, const std::vector<uint> &labels) {
    CrossEntropyLoss *loss_fn = new CrossEntropyLoss(labels);
    CrossEntropyLossContext *ctx = (CrossEntropyLossContext *)loss_fn->init();
    loss_fn->forward(ctx, this->forward(input));
    Matrix *grad = loss_fn->backward(ctx, nullptr);
    for (int i = layers.size()-1; i >= 0; -- i) {
        auto &ctx = ctxs[i];
        grad = layers[i]->backward(ctx, grad);
    }
    loss_fn->release(ctx);
    delete loss_fn;
}