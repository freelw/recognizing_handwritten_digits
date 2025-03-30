#include "ffn.h"

PositionwiseFFN::PositionwiseFFN(uint _num_hidden, uint _num_out)
    : num_hidden(_num_hidden), num_out(_num_out) {
    dense1 = new autograd::LazyLiner(num_hidden, false);
    dense2 = new autograd::LazyLiner(num_out, false);
}

PositionwiseFFN::~PositionwiseFFN() {
    delete dense1;
    delete dense2;
}

autograd::Node *PositionwiseFFN::forward(autograd::Node *x) {
    return dense2->forward(dense1->forward(x)->Relu());
}