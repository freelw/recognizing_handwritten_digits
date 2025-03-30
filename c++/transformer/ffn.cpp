#include "ffn.h"

PositionwiseFFN::PositionwiseFFN(uint _num_hidden) : num_hidden(_num_hidden) {
    dense1 = new autograd::LazyLiner(num_hidden);
    dense2 = new autograd::LazyLiner(num_hidden);
}

PositionwiseFFN::~PositionwiseFFN() {
    delete dense1;
    delete dense2;
}

autograd::Node *PositionwiseFFN::forward(autograd::Node *x) {
    return dense2->forward(dense1->forward(x)->Relu());
}