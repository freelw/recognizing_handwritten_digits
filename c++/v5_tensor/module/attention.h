#ifndef ATTENTION_H
#define ATTENTION_H

#include "graph/node.h"

class DotProductAttention {
    public:
        DotProductAttention()
        : attention_weights(nullptr) {}
        graph::Node *forward(
            graph::Node *query, graph::Node *key,
            graph::Node *value, Tensor *valid_lens
        );
    public:
        graph::Node *attention_weights;
};

#endif