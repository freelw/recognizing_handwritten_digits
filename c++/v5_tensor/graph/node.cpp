#include "node.h"
#include "actions.h"

namespace graph {

    void Node::backward() {
        assert(ref_cnt == 0);
        if (!is_require_grad()) {
            return;
        }
        assert(grad != nullptr);
        for (auto edge : edges) {
            edge->node->dec_ref();
            if (!edge->node->is_require_grad()) {
                continue;
            }
            edge->backward(grad);
        }
        for (auto edge : edges) {
            if (edge->node->is_require_grad() && 
                edge->node->get_ref() == 0) {
                edge->node->backward();
            }
        }
    }

    Node *Node::transpose() {
        Tensor *l_tensor = this->get_tensor();
        Tensor *res_tensor = l_tensor->transpose();
        Node *res_node = allocNode(res_tensor);
        if (is_require_grad()) {
            res_node->require_grad();
            res_node->edges.push_back(TransposeEdge::create(this));
        }
        return res_node;
    }

    Node *Node::expand_add(Node *rhs) {
        Tensor *res_tensor = allocTensor(t->get_shape(), "expand_add");
        Tensor *r_tensor = rhs->get_tensor();
        gCreateAction(
            new ExpandAddAction(
                this->get_tensor(),
                rhs->get_tensor(),
                res_tensor
            )
        );
        Node *res_node = allocNode(res_tensor);
        if (is_require_grad() || rhs->is_require_grad()) {
            res_node->require_grad();
            if (is_require_grad()) {
                res_node->edges.push_back(AddEdge::create(this));
            }
            if (rhs->is_require_grad()) {
                res_node->edges.push_back(ExpandAddEdge::create(rhs));
            }
        }
        return res_node;
    }

    Node *Node::at(Node *rhs) {
        Tensor *r_tensor = rhs->get_tensor();
        Tensor *l_tensor = this->get_tensor();
        assert(l_tensor->get_rank() == 2);
        assert(r_tensor->get_rank() == 2);
        assert(l_tensor->get_shape()[1] == r_tensor->get_shape()[0]);
        Tensor *res_tensor = allocTensor({l_tensor->get_shape()[0], r_tensor->get_shape()[1]}, "res_at");
        gCreateAction(
            new AtAction(
                this->get_tensor(),
                rhs->get_tensor(),
                res_tensor
            )
        );
        Node *res_node = allocNode(res_tensor);
        if (is_require_grad() || rhs->is_require_grad()) {
            res_node->require_grad();
            if (is_require_grad()) {
                res_node->edges.push_back(MatMulLEdge::create(this, rhs));
            }
            if (rhs->is_require_grad()) {
                res_node->edges.push_back(MatMulREdge::create(rhs, this));
            }
        }
        return res_node;
    }

    Node *Node::relu() {
        Tensor *l_tensor = this->get_tensor();
        Tensor *res_tensor = allocTensor(l_tensor->get_shape(), "relu_res");
        gCreateAction(
            new ReluAction(
                l_tensor,
                res_tensor
            )
        );
        Node *res_node = allocNode(res_tensor);
        if (is_require_grad()) {
            res_node->require_grad();
            res_node->edges.push_back(ReluEdge::create(this));
        }
        return res_node;
    }

    Node *Node::CrossEntropy(Tensor *labels) {
        assert(labels->get_rank() == 1);
        assert(
            labels->get_dtype() == INT32 
        );
        assert(labels->get_shape()[0] == this->get_tensor()->get_shape()[0]);
        Tensor *tensor_maxs = allocTensor(labels->get_shape(), "maxs");
        Tensor *tensor_sums = allocTensor(labels->get_shape(), "sums");
        Tensor *ce_res = allocTensor({1}, "cross_entropy");

        gCreateAction(
            new CrossEntropyAction(
                this->get_tensor(),
                labels,
                tensor_maxs,
                tensor_sums,
                ce_res
            )
        );
        Node *res_node = allocNode(ce_res);
        if (is_require_grad() ) {
            res_node->require_grad();
            res_node->edges.push_back(CrossEntropyEdge::create(this, labels, tensor_maxs, tensor_sums));
        }
        return res_node;
    }

    void CrossEntropyEdge::backward(Tensor *) {
        gCreateAction(
            new CrossEntropyBackwardAction(
                node->get_tensor(),
                labels,
                maxs,
                sums,
                node->get_grad()
            )
        );
    }

    std::vector<Edge *> edges;
    std::vector<Node *> nodes;

    Node *allocNode(Tensor *t) {
        Node *node = new Node(t);
        nodes.push_back(node);
        return node;
    }

    void gAddEdge(Edge *edge) {
        edges.push_back(edge);
    }

    void freeAllNodes() {
        for (Node *node : nodes) {
            delete node;
        }
        nodes.clear();
    }

    void freeAllEdges() {
        for (Edge *edge : edges) {
            delete edge;
        }
        edges.clear();
    }
} // namespace graph