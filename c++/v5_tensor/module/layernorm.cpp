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
    return x->norm()->expand_mul(gamma)->expand_add(beta);
}

std::vector<Parameter*> LayerNorm::parameters() {
    std::vector<Parameter*> params;
    params.push_back(Pgamma);
    params.push_back(Pbeta);
    return params;
}