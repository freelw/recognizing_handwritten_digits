#ifndef POSITIONWISE_FFN_H
#define POSITIONWISE_FFN_H

#include "autograd/node.h"
#include "dropout.h"
#include "liner.h"

class PositionwiseFFN {
    public:
        PositionwiseFFN(u_int32_t _num_hidden);
        ~PositionwiseFFN();
        autograd::Node *forward(autograd::Node *x);
    private:
        uint num_hidden;
        autograd::LazyLiner *dense1;
        autograd::LazyLiner *dense2;
};

#endif