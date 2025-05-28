#include "attention.h"
#include <cmath>

extern bool g_training;

DotProductAttention::DotProductAttention(float p)
    :  attention_weights(nullptr) {
    dropout = new Dropout(p);
}

DotProductAttention::~DotProductAttention() {
    assert(dropout != nullptr);
    delete dropout;
}

graph::Node *DotProductAttention::forward(
    graph::Node *query, graph::Node *key,
    graph::Node *value, Tensor *valid_lens
) {

    assert(query->get_tensor()->get_dim() == 3);
    assert(key->get_tensor()->get_dim() == 3);
    assert(value->get_tensor()->get_dim() == 3);
    if (valid_lens != nullptr) {
        assert(valid_lens->get_dtype() == INT32);
    }
    
    auto q_shape = query->get_tensor()->get_shape();
    auto k_shape = key->get_tensor()->get_shape();
    auto v_shape = value->get_tensor()->get_shape();

    // batch size eq
    assert(q_shape[0] == k_shape[0]);
    assert(q_shape[0] == v_shape[0]);
    // q k d eq
    assert(q_shape[2] == k_shape[2]);
    // k v validate
    assert(k_shape[1] == v_shape[1]);

    auto q_dim = query->get_tensor()->get_dim();
    assert(q_dim == 3);
    auto d = q_shape[q_dim-1];

    attention_weights = query->bmm(key->transpose(1, 2))
        ->div(std::sqrt(static_cast<float>(d)))
        ->masked_softmax(valid_lens);

    auto dropout_attention_weights = attention_weights;
    if (g_training) {
        dropout_attention_weights = dropout
            ->forward(dropout_attention_weights);
    }
    return attention_weights->bmm(value);
}