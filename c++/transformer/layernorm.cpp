#include "layernorm.h"


LayerNorm::LayerNorm(uint dim) {
    mgamma = new Matrix(Shape(dim, 1));
    mbeta = new Matrix(Shape(dim, 1));
    gamma = new autograd::Node(mgamma, true);
    beta = new autograd::Node(mbeta, true);
    pgamma = new autograd::Parameters(gamma);
    pbeta = new autograd::Parameters(beta);
}

LayerNorm::~LayerNorm() {
    delete mgamma;
    delete mbeta;
    delete gamma;
    delete beta;
    delete pgamma;
    delete pbeta;
}

autograd::Node* LayerNorm::forward(autograd::Node* x) {
    return gamma->at(x->Norm())->expand_add(beta);
}