#include "addnorm.h"

AddNorm::AddNorm(int _num_hidden, DATATYPE _dropout)
    : num_hidden(_num_hidden),
    training(true),
    dropout(_dropout),
    dropout_layer(nullptr),
    layernorm(nullptr)
{
    if (dropout > 0) {
        dropout_layer = new autograd::Dropout(dropout);
    }
    layernorm = new LayerNorm(num_hidden);
}

AddNorm::~AddNorm() {
    if (dropout > 0 && dropout_layer != nullptr) {
        delete dropout_layer;
    }
    delete layernorm;
}

autograd::Node *AddNorm::forward(autograd::Node *x, autograd::Node *y) {
    assert(x->checkShape(y->getShape()));
    autograd::Node * _y;
    if (dropout > 0 && is_training()) {
        _y = dropout_layer->forward(y);
    } else {
        _y = y;
    }
    return layernorm->forward((*x) + _y);
}
