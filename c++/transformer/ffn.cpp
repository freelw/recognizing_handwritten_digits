#include "ffn.h"

PositionwiseFFN::PositionwiseFFN(uint _num_hidden, uint _num_out)
    : num_hidden(_num_hidden), num_out(_num_out) {
    dense1 = new autograd::LazyLinear(num_hidden, true);
    dense2 = new autograd::LazyLinear(num_out, true);
    #pragma message("dense bias should be random, fix later.")
}

PositionwiseFFN::~PositionwiseFFN() {
    delete dense1;
    delete dense2;
}

autograd::Node *PositionwiseFFN::forward(autograd::Node *x) {
    return dense2->forward(dense1->forward(x)->Relu());
}