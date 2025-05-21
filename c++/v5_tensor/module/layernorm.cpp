#include "layernorm.h"

LayerNorm::LayerNorm(int dim) {
    Tensor *t_gamma = allocTensor({dim}, "layernorm_gamma");
    Tensor *t_beta = allocTensor({dim}, "layernorm_beta");
    gamma = graph::allocNode(t_gamma);
    beta = graph::allocNode(t_beta);
    gamma->require_grad();
    beta->require_grad();
    Pgamma = allocParameter(gamma);
    Pbeta = allocParameter(beta);
}

graph::Node* LayerNorm::forward(graph::Node *x) {
    assert(x->get_tensor()->get_dim() >= 2);
    auto origin_shape = x->get_tensor()->get_shape();
    auto dim = x->get_tensor()->get_dim();
    x = x->reshape({-1, origin_shape[dim - 1]});
    x = x->norm()->expand_mul(gamma)->expand_add(beta);
    return x->reshape(origin_shape);
}

std::vector<Parameter*> LayerNorm::parameters() {
    std::vector<Parameter*> params;
    params.push_back(Pgamma);
    params.push_back(Pbeta);
    return params;
}