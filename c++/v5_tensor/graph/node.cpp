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

    Node *Node::expand_add(Node *rhs) {
        Tensor *res_tensor = allocTensor(t->get_shape());
        Tensor *r_tensor = rhs->get_tensor();
        gCreateAction(
            new AddAction(
                this->get_tensor(),
                rhs->get_tensor(),
                res_tensor
            )
        );
        Node *res_node = allocNode(res_tensor);
        res_node->edges.push_back(AddEdge::create(this));
        res_node->edges.push_back(ExpandAddEdge::create(rhs));
        return res_node;
    }

    Node *Node::at(Node *rhs) {
        return nullptr;
    }

    Node *Node::relu() {
        return nullptr;
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