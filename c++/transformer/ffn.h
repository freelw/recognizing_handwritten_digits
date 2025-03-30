#ifndef POSITIONWISE_FFN_H
#define POSITIONWISE_FFN_H

#include "autograd/node.h"
#include "dropout.h"
#include "liner.h"

class PositionwiseFFN {
    public:
        PositionwiseFFN(uint _num_hidden, uint _num_out);
        ~PositionwiseFFN();
        autograd::Node *forward(autograd::Node *x);
    private:
        uint num_hidden;
        uint num_out;
        autograd::LazyLiner *dense1;
        autograd::LazyLiner *dense2;
};

#endif