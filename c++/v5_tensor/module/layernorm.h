#ifndef LAYERNORM_H
#define LAYERNORM_H


#include "graph/node.h"
#include "optimizers/parameter.h"
class LayerNorm {
    public:
        LayerNorm(int dim);
        ~LayerNorm() = default;
        graph::Node* forward(graph::Node *x);
        std::vector<Parameter*> parameters();
    private:
        graph::Node *gamma;
        graph::Node *beta;
        Parameter *Pgamma;
        Parameter *Pbeta;
};
#endif