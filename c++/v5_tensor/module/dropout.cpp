#include "dropout.h"
extern bool g_training;

graph::Node *Dropout::forward(graph::Node *x) {
    if (g_training) {
        graph::Node *res_node = nullptr;
        Tensor *new_tensor = allocTensor(
            x->get_tensor()->get_shape(),
            x->get_tensor()->get_name() + "dropout",
            x->get_tensor()->get_dtype()
        );
        gCreateAction(
            new DropoutAction(
                new_tensor,
                x->get_tensor(),
                p
            )
        );
        if (x->is_require_grad()) {
            Tensor *new_grad = allocTensor(
                x->get_grad()->get_shape(),
                x->get_grad()->get_name() + "dropout_grad",
                x->get_grad()->get_dtype()
            );
            res_node = graph::allocNode(new_tensor, new_grad);
            res_node->edges.push_back(graph::DropoutEdge::create(x));
        } else {
            res_node = graph::allocNode(new_tensor);
        }
        return res_node;
    } else {
        return x;
    }
}