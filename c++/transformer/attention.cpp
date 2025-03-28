#include "attention.h"



DotProductAttetion::DotProductAttetion(DATATYPE _dropout)
    : dropout(_dropout), dropout_layer(nullptr) {
    assert (dropout >= 0 && dropout < 1);
    if (dropout > 0) {
        dropout_layer = new autograd::Dropout(dropout);
    }
}

DotProductAttetion::~DotProductAttetion() {
    if (dropout > 0 && dropout_layer != nullptr) {
        delete dropout_layer;
    }
}

std::vector<autograd::Node *> DotProductAttetion::forward(
    const std::vector<autograd::Node *> &Q,
    const std::vector<autograd::Node *> &K,
    const std::vector<autograd::Node *> &V
) {

    assert (Q.size() == K.size() && K.size() == V.size());
    std::vector<autograd::Node *> res;
    for (size_t i = 0; i < Q.size(); i++) {
        autograd::Node *q = Q[i];
        autograd::Node *k = K[i];
        autograd::Node *v = V[i];
        autograd::Node *score = q->at(k->Transpose());
        // score = score->div(sqrt(k->shape()[0]));
        // score = score->softmax(0);
        // if (dropout > 0) {
        //     score = dropout_layer->forward(score);
        // }
        // autograd::Node *att = score->dot(v);
        // res.push_back(att);
    }
    return res;

}