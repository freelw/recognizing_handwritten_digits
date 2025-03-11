#include "node.h"
#include <assert.h>

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
                node->edges.push_back(allocEdge({}, ExpandAdd, this));
            }
            if (rhs->is_require_grad()) {
                node->edges.push_back(allocEdge({}, ExpandAdd, rhs));
            }
        }
        return node;
    }

    Node *Node::at(Node *rhs) {
        auto *node = allocNode(w->at(*(rhs->w)));
        if (is_require_grad() || rhs->is_require_grad()) {
            node->require_grad();
            if (is_require_grad()) {
                node->edges.push_back(allocEdge({rhs->w}, MatMul, this));
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
            node->edges.push_back(allocEdge({w}, OpType::Relu, this));
        }
        return node;
    }

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
            switch (edge->type) {
                case Add:
                    assert(edge->params.size() == 0);
                    *edge->node->grad += *grad;
                    break;
                case ExpandAdd:
                    assert(edge->params.size() == 0);
                    *edge->node->grad += *(grad->sum(1));
                    break;
                case MatMul:
                    assert(edge->params.size() == 1);
                    *edge->node->grad += *(grad->at(*(edge->params[0]->transpose())));
                    break;
                case OpType::Relu:
                    assert(edge->params.size() == 0);
                    break;
                default:
                    break;
            }
        }
        for (auto edge : edges) {
            if (edge->node->is_require_grad() && 
                edge->node->get_ref() == 0) {
                edge->node->backward();
            }
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