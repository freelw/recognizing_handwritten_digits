#include "attention.h"

DotProductAttetion::DotProductAttetion(DATATYPE _dropout)
    : dropout(_dropout), dropout_layer(nullptr), is_training(true) {
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
    std::vector<autograd::Node *> scores;
    for (size_t i = 0; i < Q.size(); i++) {
        autograd::Node *q = Q[i];
        autograd::Node *k = K[i];
        autograd::Node *score = q->Transpose()->at(k)->Transpose();
        score = score->Div(sqrt(k->getShape().rowCnt));
        score = score->Softmax();
        scores.push_back(score);
    }

    if (dropout > 0 && training()) {
        scores = dropout_layer->forward(scores);
    }

    for (size_t i = 0; i < V.size(); i++) {
        autograd::Node *att = V[i]->at(scores[i]);
        res.push_back(att);
    }
    return res;
}