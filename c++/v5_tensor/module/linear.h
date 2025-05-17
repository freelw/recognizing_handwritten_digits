#ifndef LINEAR_H
#define LINEAR_H

#include "optimizers/parameter.h"

enum ACTIVATION {
    RELU,
    SIGMOID,
    TANH,
    NONE
};

class Linear {
    public:
        Linear(int input_num, int output_num, ACTIVATION act, bool _bias = true);
        ~Linear() = default;
        graph::Node *forward(graph::Node *input);
        std::vector<Parameter *> get_parameters();
    private:
        bool bias;
        graph::Node *w;
        graph::Node *b;
        Parameter *Pw;
        Parameter *Pb;
};

#endif