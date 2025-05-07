#ifndef PARAMETER_H
#define PARAMETER_H


#include "tensor/tensor.h"
#include "graph/node.h"

class Parameter {
    public:
        Parameter(graph::Node *_node);
        Tensor *get_grad();
        bool is_require_grad();
    private:
        graph::Node *node;
        Tensor *m;
        Tensor *v;
        int t;
};
#endif