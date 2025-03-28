#ifndef ATTENTION_H
#define ATTENTION_H

#include "autograd/node.h"
#include "dropout.h"


class DotProductAttetion {
    public:
        DotProductAttetion(DATATYPE _dropout);
        ~DotProductAttetion();
        std::vector<autograd::Node *> forward(
            const std::vector<autograd::Node *> &Q,
            const std::vector<autograd::Node *> &K,
            const std::vector<autograd::Node *> &V
        );
    private:
        DATATYPE dropout;
        autograd::Dropout *dropout_layer;
};

#endif