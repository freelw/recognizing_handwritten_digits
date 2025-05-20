#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "graph/node.h"
#include "optimizers/parameter.h"

class Embedding {
    public:
        Embedding(int _vocab_size, int _hidden_num);
        ~Embedding() = default;
        graph::Node *forward(Tensor *indices);
        std::vector<Parameter *> get_parameters();
    private:
        int vocab_size;
        int hidden_num;
        graph::Node *w;
        Parameter *Pw;
};

#endif