#include "node.h"
#include <assert.h>

namespace autograd {

    Matrix *CrossEntropyLoss(Matrix *input, const std::vector<uint> &labels, std::vector<CrosEntropyInfo> &info) { 
        assert(input->getShape().colCnt == labels.size());
        assert(info.size() == 0);
        Matrix *loss = allocTmpMatrix(Shape(1,1));
        DATATYPE loss_value = 0;
        for (uint j = 0; j < input->getShape().colCnt; ++ j) {
            DATATYPE max = (*input)[0][j];
            for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
                auto e = (*input)[i][j];
                if (max < e) {
                    max = e;
                }
            }
            DATATYPE sum = 0;
            auto target = labels[j];
            DATATYPE zt = (*input)[target][j];
            for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
                DATATYPE e = (*input)[i][j];
                e = std::exp(e-max);
                sum += e;
            }
            CrosEntropyInfo p;
            p.sum = sum;
            p.max = max;
            info.push_back(p);
            loss_value += -(zt - max - log(sum));
        }
        (*loss)[0][0] = loss_value/labels.size();
        return loss;
    }

    Matrix * CrossEntropyGrad(Matrix *input, const std::vector<uint> &labels) {
        Matrix *grad = allocTmpMatrix(Shape(input->getShape()));
        auto batch_size = labels.size();
        for (uint j = 0; j < input->getShape().colCnt; ++ j) {
            DATATYPE max = (*input)[0][j];
            for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
                auto e = (*input)[i][j];
                if (max < e) {
                    max = e;
                }
            }
            DATATYPE sum = 0;
            auto target = labels[j];
            DATATYPE zt = (*input)[target][j];
            for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
                DATATYPE e = (*input)[i][j];
                e = std::exp(e-max);
                sum += e;
            }
            for (uint i = 0; i < input->getShape().rowCnt; ++i) {
                if (i == target) {
                    continue;
                }
                auto &_grad = (*grad)[i][j];
                _grad = std::exp((*input)[i][j] - max) / sum / batch_size;
            }
            (*grad)[target][j] = (std::exp((*input)[target][j] - max) / sum - 1) / batch_size;
        }
        return grad;
    }
   
    Node *Node::operator+(Node *rhs) {
        auto *node = allocNode(*w + *(rhs->w));
        if (is_require_grad() || rhs->is_require_grad()) {
            node->require_grad();
            if (is_require_grad()) {
                node->edges.push_back(AddEdge::create(this));
            }
            if (rhs->is_require_grad()) {
                node->edges.push_back(AddEdge::create(rhs));
            }
        }
        return node;
    }

    Node *Node::expand_add(Node *rhs) {
        auto *node = allocNode(w->expand_add(*(rhs->w)));
        if (is_require_grad() || rhs->is_require_grad()) {
            node->require_grad();
            if (is_require_grad()) {
                node->edges.push_back(ExpandAddEdge::create(this));
            }
            if (rhs->is_require_grad()) {
                node->edges.push_back(ExpandAddEdge::create(rhs));
            }
        }
        return node;
    }

    Node *Node::at(Node *rhs) {
        auto *node = allocNode(w->at(*(rhs->w)));
        if (is_require_grad() || rhs->is_require_grad()) {
            node->require_grad();
            if (is_require_grad()) {
                node->edges.push_back(MatMulEdge::create(this, rhs->get_weight()));
            }
            if (rhs->is_require_grad()) {
                node->edges.push_back(MatMulEdge::create(rhs, w));
            }
        }
        return node;
    }

    Node *Node::Relu() {
        auto *node = allocNode(w->Relu());
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(ReluEdge::create(this));
        }
        return node;
    }

    Node *Node::CrossEntropy(const std::vector<uint> &labels) {
        std::vector<CrosEntropyInfo> info;
        auto *node = allocNode(::autograd::CrossEntropyLoss(w, labels, info));
        assert(info.size() == w->getShape().colCnt);
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(CrossEntropyEdge::create(this, labels));
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
            edge->backward(grad);
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