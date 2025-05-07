#include "parameter.h"

Parameter::Parameter(graph::Node *_node)
    : node(_node), t(0) {
    Tensor *t = _node->get_tensor();
    m = allocTensor(t->get_shape(), t->get_name()+"_m");
    v = allocTensor(t->get_shape(), t->get_name()+"_v");
}

Tensor *Parameter::get_w() {
    return node->get_tensor();
}

Tensor *Parameter::get_grad() {
    assert(node->is_require_grad());
    assert(node->get_grad() != nullptr);
    return node->get_grad();
}

Tensor *Parameter::get_m() {
    return m;
}

Tensor *Parameter::get_v() {
    return v;
}

bool Parameter::is_require_grad() {
    return node->is_require_grad();
}

std::vector<Parameter *> g_parameters;

Parameter *allocParameter(graph::Node *_node) {
    Parameter *param = new Parameter(_node);
    g_parameters.push_back(param);
    return param;
}

void releaseParameters() {
    for (auto param : g_parameters) {
        delete param;
    }
    g_parameters.clear();
}