#include "embedding.h"

Embedding::Embedding(int _vocab_size, int _hidden_num)
    : vocab_size(_vocab_size), hidden_num(_hidden_num) {
    Tensor * t = allocTensor({vocab_size, hidden_num}, "embedding");
    w = graph::allocNode(t);
    w->require_grad();
    Pw = allocParameter(w);
}

graph::Node *Embedding::forward(Tensor *indices) {
    assert(indices->get_dim() == 1);
    Tensor *res = allocTensor({indices->get_shape()[0], hidden_num}, "embedding_out");

    auto res_node = graph::allocNode(res);
    return res_node;
}