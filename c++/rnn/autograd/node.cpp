#include "node.h"


namespace autograd {
   
    Node *Node::operator+(Node *rhs) {
        auto *node = allocNode(*w + *(rhs->w));
        if (is_require_grad() || rhs->is_require_grad()) {
            node->require_grad();
            if (is_require_grad()) {
                node->edges.push_back(allocEdge({}, Add, this));
            }
            if (rhs->is_require_grad()) {
                node->edges.push_back(allocEdge({}, Add, rhs));
            }
        }
        return node;
    }

    Node *Node::expand_add(Node *rhs) {
        auto *node = allocNode(w->expand_add(*(rhs->w)));
        if (is_require_grad() || rhs->is_require_grad()) {
            node->require_grad();
            if (is_require_grad()) {
                node->edges.push_back(allocEdge({}, Add, this));
            }
            if (rhs->is_require_grad()) {
                node->edges.push_back(allocEdge({}, Add, rhs));
            }
        }
        return node;
    }

    Node *Node::at(Node *rhs) {
        auto *node = allocNode(w->at(*(rhs->w)));
        if (is_require_grad() || rhs->is_require_grad()) {
            node->require_grad();
            if (is_require_grad()) {
                node->edges.push_back(allocEdge({rhs->w}, MatMul, this->grad));
            }
            if (rhs->is_require_grad()) {
                node->edges.push_back(allocEdge({w}, MatMul, rhs));
            }
        }
        return node;
    }

    Node *Node::Relu() {
        auto *node = allocNode(w->Relu());
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(allocEdge({w}, OpType::Relu, grad));
        }
        return node;
    }

    void Node::backward() {
        assert(ref_cnt == 0);
        if (!is_require_grad()) {
            return;
        }
        if (grad == nullptr) {
            grad = allocTmpMatrix(w->getShape());
        }
        for (auto edge : edges) {
            switch (edge->type) {
                case Add:
                    *grad += *edge->t_grad;
                    break;
                case MatMul:
                    *grad += *(*edge->t_grad * *(edge->params[0]->transpose()));
                    break;
                case OpType::Relu:
                    *grad += *(*edge->t_grad * *(w->Relu_prime()));
                    break;
                default:
                    break;
            }
        }
        for (auto edge : edges) {
            edge->node->backward();
        }
    }

    std::vector<Edge *> edges;
    std::vector<Node *> nodes;

    Edge *allocEdge(const std::vector<Matrix *> &_params, OpType _type, Node *node) {
        Edge *edge = new Edge(_params, _type, node);
        edges.push_back(edge);
        return edge;
    }

    Node *allocNode(Matrix *w) {
        Node *node = new Node(w, true);
        nodes.push_back(node);
        return node;
    }

    void freeAllNodes() {
        for (auto node : nodes) {
            delete node;
        }
        nodes.clear();
    }

    void freeAllEdges() {
        for (auto edge : edges) {
            delete edge;
        }
        edges.clear();
    }
} // namespace autograd