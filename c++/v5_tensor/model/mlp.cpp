#include "mlp.h"
#include "graph/node.h"


MLP::MLP(int32_t _input, const std::vector<int32_t> &_outputs) {
    w1 = allocTensor({ _input, _outputs[0] }, "w1");
    w2 = allocTensor({ _outputs[0], _outputs[1] }, "w2");
    bias1 = allocTensor({ _outputs[0] }, "bias1");
    bias2 = allocTensor({ _outputs[1] }, "bias2");

    nw1 = graph::allocNode(w1);
    nw2 = graph::allocNode(w2);
    nb1 = graph::allocNode(bias1);
    nb2 = graph::allocNode(bias2);

    nw1->require_grad();
    nw2->require_grad();
    nb1->require_grad();
    nb2->require_grad();

    nw1->init_weight_gauss(0.02, 0);
    nw2->init_weight_gauss(0.02, 0);

    pw1 = allocParameter(nw1);
    pw2 = allocParameter(nw2);
    pb1 = allocParameter(nb1);
    pb2 = allocParameter(nb2);
}

std::vector<Parameter*> MLP::get_parameters() {
    return { pw1, pw2, pb1, pb2 };
}

graph::Node *MLP::forward(graph::Node *input) {
    graph::Node *x = input->at(nw1);
    x = x->expand_add(nb1);
    x = x->relu();
    x = x->at(nw2);
    x = x->expand_add(nb2);
    return x;
}