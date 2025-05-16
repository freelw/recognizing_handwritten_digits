#include "attention.h"
#include <cmath>

graph::Node *DotProductAttention::forward(
    graph::Node *query, graph::Node *key,
    graph::Node *value, Tensor *valid_lens
) {
    auto q_shape = query->get_tensor()->get_shape();
    auto q_length = query->get_tensor()->length();
    auto d = q_shape[q_length-1];

    auto scores = query->bmm(key->transpose(1, 2))
        ->div(std::sqrt(static_cast<float>(d)));
    attention_weights = scores->masked_softmax(valid_lens);
    // todo : dropout
    return attention_weights->bmm(value);
}