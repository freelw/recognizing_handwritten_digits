#include "parameter.h"

Parameter::Parameter(graph::Node *_node)
    : node(_node), t(0) {
    Tensor *t = _node->get_tensor();
    m = allocTensor(t->get_shape(), t->get_name()+"_m");
    v = allocTensor(t->get_shape(), t->get_name()+"_v");
}