#include "models.h"

MLP::MLP(uint _input, const std::vector<uint> &_outputs, bool _rand)
    : input(_input), outputs(_outputs), rand(_rand)
{

}

void MLP::init() {
    uint x = input;
    for (uint i = 0; i < outputs.size(); ++ i) {
        uint y = outputs[i];
        layers.push_back(new Liner(x, y, rand));
        x = y;
        if (i < outputs.size()-1) {
            layers.push_back(new Relu());
        }
    }

    for (auto &layer : layers) {
        ctxs.push_back(layer->init());
    }

    assert(layers.size() == 3); // 2 liner 1 relu
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

DATATYPE MLP::backward(Matrix *input, const std::vector<uint> &labels) {
    CrossEntropyLoss *loss_fn = new CrossEntropyLoss(labels);
    CrossEntropyLossContext *ctx = (CrossEntropyLossContext *)loss_fn->init();
    auto loss = loss_fn->forward(ctx, this->forward(input));
    loss->checkShape(Shape(1, 1));
    Matrix *grad = loss_fn->backward(ctx, nullptr);
    for (int i = layers.size()-1; i >= 0; -- i) {
        auto &ctx = ctxs[i];
        grad = layers[i]->backward(ctx, grad);
    }
    loss_fn->release(ctx);
    delete loss_fn;
    return (*loss)[0][0];
}

std::vector<Parameters*> MLP::get_parameters() {
    std::vector<Parameters*> res;
    for (auto layer : layers) {
        auto parameters = layer->get_parameters();
        res.insert(res.end(), parameters.begin(), parameters.end());
    }
    return res;
}

void MLP::zero_grad() {
    for (auto layer : layers) {
        layer->zero_grad();
    }
}