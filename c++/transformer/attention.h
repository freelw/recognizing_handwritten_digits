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
            const std::vector<autograd::Node *> &V,
            const std::vector<uint> &valid_lens
        );
        void train(bool _is_training) {
            is_training = _is_training;
        }
        bool training() {
            return is_training;
        }
    private:
        DATATYPE dropout;
        autograd::Dropout *dropout_layer;
        bool is_training;
};

#endif