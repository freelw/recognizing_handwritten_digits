#include "embedding.h"
#include <cmath>

Embedding::Embedding(int _vocab_size, int _hidden_num, bool const_weight)
    : vocab_size(_vocab_size), hidden_num(_hidden_num) {
    Tensor * t = allocTensor({vocab_size, hidden_num}, "embedding");
    w = graph::allocNode(t);
    w->require_grad();
    if (const_weight) {
        w->init_weight_for_dbg(10000.0f);
    } else {
        w->init_weight_uniform(std::sqrt(1.0/hidden_num));
    }
    Pw = allocParameter(w);
}

graph::Node *Embedding::forward(Tensor *indices) {
    assert(indices->get_dim() == 1);
    Tensor *res = allocTensor({indices->get_shape()[0], hidden_num}, "embedding_out");
    auto res_node = graph::allocNode(res);
    res_node->require_grad();
    gCreateAction(
        new EmbeddingAction(
            w->get_tensor(),
            indices,
            res
        )
    );
    res_node->edges.push_back(
        new graph::EmbeddingEdge(w, indices)
    );
    return res_node;
}