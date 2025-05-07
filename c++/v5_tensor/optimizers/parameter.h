#ifndef PARAMETER_H
#define PARAMETER_H


#include "tensor/tensor.h"
#include "graph/node.h"

class Parameter {
    public:
        Parameter(graph::Node *_node);
    private:
        graph::Node *node;
        Tensor *m;
        Tensor *v;
        int t;
};
#endif