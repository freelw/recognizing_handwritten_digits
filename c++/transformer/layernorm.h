#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "autograd/node.h"
#include "autograd/parameters.h"

class Norm {
    public:
        Norm();
        ~Norm();
        autograd::Node* forward(autograd::Node* x);
};

class LayerNorm {
    public:
        LayerNorm(uint dim);
        ~LayerNorm();
        autograd::Node* forward(autograd::Node* x);
        std::vector<autograd::Parameters*> parameters();
    private:
        Norm norm;
        Matrix *mgamma;
        Matrix *mbeta;
        autograd::Node *gamma;
        autograd::Node *beta;
        autograd::Parameters *pgamma;
        autograd::Parameters *pbeta;
};
#endif