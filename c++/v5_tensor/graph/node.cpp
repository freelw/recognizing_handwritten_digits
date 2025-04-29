#include "node.h"

namespace graph {

    void Node::backward() {
        
    }

    // Node *operator+(Node *rhs);
    // Node *operator+=(Node *rhs);
    // Node *operator*(Node *rhs);
    // Node *expand_add(Node *rhs);
    // Node *at(Node *rhs);
    // Node *relu();
    // Node *transpose_2d();

    Node *Node::operator+(Node *rhs) {
        Tensor *res = allocTensor(t->get_shape());
        gCreadeAction(new AddAction(this, rhs, res));
    }

    std::vector<Edge *> edges;
    std::vector<Node *> nodes;

    Node *allocNode(Tensor *t) {
        Node *node = new Node(t);
        nodes.push_back(node);
        return node;
    }
} // namespace graph