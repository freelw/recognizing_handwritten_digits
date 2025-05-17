#include "dropout.h"
extern bool g_training;

graph::Node *Dropout::forward(graph::Node *x) {
    if (g_training) {
        auto x_tensor = x->get_tensor();
        assert(x_tensor->get_dim() == 1);
        auto shape = x_tensor->get_shape();
        graph::Node *res_node = nullptr;
        Tensor *mask = allocTensor(
            shape,
            x_tensor->get_name() + "_dropout_mask",
            x_tensor->get_dtype()
        );
        gCreateAction(
            new DropoutMaskAction(
                mask,
                p
            )
        );
        Tensor *res_tensor = allocTensor(
            shape,
            x_tensor->get_name() + "_dropout_res",
            x_tensor->get_dtype()
        );

        gCreateAction(
            new MulAction(
                x_tensor,
                mask,
                res_tensor
            )
        );
        if (x->is_require_grad()) {
            Tensor *res_grad = allocTensor(
                x->get_grad()->get_shape(),
                x->get_grad()->get_name() + "dropout_grad",
                x->get_grad()->get_dtype()
            );
            res_node = graph::allocNode(res_tensor, res_grad);
            res_node->edges.push_back(graph::DropoutEdge::create(x, mask));
        } else {
            res_node = graph::allocNode(res_tensor);
        }
        return res_node;
    } else {
        return x;
    }
}