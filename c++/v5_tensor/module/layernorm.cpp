#include "layernorm.h"

LayerNorm::LayerNorm(int dim) {

    Tensor *t_gamma = allocTensor({1, dim}, "layernorm_gamma");
    Tensor *t_beta = allocTensor({1, dim}, "layernorm_beta");

    gamma = graph::allocNode(t_gamma);
    beta = graph::allocNode(t_beta);

    gamma->require_grad();
    beta->require_grad();

    Pgamma = allocParameter(gamma);
    Pbeta = allocParameter(beta);
}

LayerNorm::~LayerNorm() {

}

graph::Node* LayerNorm::forward(graph::Node* x) {
    assert(false);
    return nullptr;
}

std::vector<Parameter*> LayerNorm::parameters() {
    std::vector<Parameter*> params;
    params.push_back(Pgamma);
    params.push_back(Pbeta);
    return params;
}