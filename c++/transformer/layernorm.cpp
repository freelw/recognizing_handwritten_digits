#include "layernorm.h"


Norm::Norm() {
    
}

Norm::~Norm() {
    
}

autograd::Node* Norm::forward(autograd::Node* x) {
    
}

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
    auto node = gamma->at(x)->expand_add(beta);
    return norm.forward(node);
}