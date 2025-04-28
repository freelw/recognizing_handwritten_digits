#include "node.cuh"
#include <assert.h>
#include <iostream>
#include <string.h>
#include "backends/ops.cuh"

namespace autograd_cuda {

    Matrix *CrossEntropyLoss(Matrix *input, const std::vector<uint> &labels, Matrix *&maxs, Matrix *&sums) { 
        return g_backend_ops->CrossEntropyLoss(input, labels, maxs, sums);
    }

    Matrix *CrossEntropyLossMask(
        Matrix *input,
        const std::vector<uint> &labels,
        std::vector<CrosEntropyInfo> &info,
        const std::vector<bool> &mask) {
        return g_backend_ops->CrossEntropyLossMask(input, labels, info, mask);
    }

    Node *Node::Norm() {
        auto avg_res = w->avg();
        auto var_res = w->var();
        DATATYPE eps = 1e-5;
        auto tmp = g_backend_ops->Norm(w, avg_res, var_res, eps);
        auto *node = allocNode(tmp);
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(NormEdge::create(this, tmp, avg_res, var_res, eps));
        }
        return node;
    }

    Node *Node::Softmax() {
        auto tmp = g_backend_ops->Softmax(w);
        auto *node = allocNode(tmp);
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(SoftmaxEdge::create(this, node));
        }
        return node;
    }

    Node *Node::Transpose() {
        auto *tmp = allocTmpMatrix(w->transpose());
        auto *node = allocNode(tmp);
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(TransposeEdge::create(this));
        }
        return node;
    }

    Node *Node::Mul(DATATYPE v) {
        auto *tmp = allocTmpMatrix(*w * v);
        auto *node = allocNode(tmp);
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(MulSingleValueEdge::create(this, v));
        }
        return node;
    }

    Node *Node::Div(DATATYPE v) {
        auto *tmp = allocTmpMatrix(*w / v);
        auto *node = allocNode(tmp);
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(DivEdge::create(this, v));
        }
        return node;
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

    Node *Node::operator*(Node *rhs) {
        auto *node = allocNode(*w * *(rhs->w));
        if (is_require_grad() || rhs->is_require_grad()) {
            node->require_grad();
            if (is_require_grad()) {
                node->edges.push_back(MulEdge::create(this, rhs->get_weight()));
            }
            if (rhs->is_require_grad()) {
                node->edges.push_back(MulEdge::create(rhs, w));
            }
        }
        return node;
    }

    Node *Node::expand_add(Node *rhs) {
        auto *node = allocNode(w->expand_add(*(rhs->w)));
        if (is_require_grad() || rhs->is_require_grad()) {
            node->require_grad();
            if (is_require_grad()) {
                node->edges.push_back(AddEdge::create(this));
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
                node->edges.push_back(MatMulLEdge::create(this, rhs->get_weight()));
            }
            if (rhs->is_require_grad()) {
                node->edges.push_back(MatMulREdge::create(rhs, w));
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
        assert(w->getShape().colCnt == labels.size());
        Matrix *maxs = nullptr;
        Matrix *sums = nullptr;
        auto *node = allocNode(::autograd_cuda::CrossEntropyLoss(w, labels, maxs, sums));
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(CrossEntropyEdge::create(this, labels, maxs, sums));
        }
        return node;
    }

    Node *Node::CrossEntropyMask(const std::vector<uint> &labels, const std::vector<bool> &mask) {
        assert(w->getShape().colCnt == labels.size());
        assert(w->getShape().colCnt == mask.size());
        std::vector<CrosEntropyInfo> info;
        auto *node = allocNode(::autograd_cuda::CrossEntropyLossMask(w, labels, info, mask));
        assert(info.size() == w->getShape().colCnt);
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(CrossEntropyMaskEdge::create(this, labels, mask, info));
        }
        return node;
    }

    Node *Node::Tanh() {
        auto *node = allocNode(w->tanh());
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(TanhEdge::create(this));
        }
        return node;
    }

    Node *Node::Sigmoid() {
        auto *node = allocNode(w->sigmoid());
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(SigmoidEdge::create(this));
        }
        return node;
    }

    Node *operator-(DATATYPE v, Node &rhs) {
        auto node = allocNode(v - *(rhs.get_weight()));
        if (rhs.is_require_grad()) {
            node->require_grad();
            node->edges.push_back(MinusEdge::create(&rhs));
        }
        return node;
    }

    Node *cat0(const std::vector<Node *> &nodes) {
        std::cerr << "cat0 not implemented" << std::endl;
        assert(false);
        return nullptr;
    }

    Node *cat1(const std::vector<Node *> &nodes) {
        std::cerr << "cat1 not implemented" << std::endl;
        assert(false);
        return nullptr;
    }

    Node *cat(const std::vector<Node *> &nodes, uint dim) {
        assert(dim == 0 || dim == 1);
        assert(nodes.size() > 0);
        if (dim == 0) { // 这里似乎反了，将错就错，dim == 0 时我们拼接行，split也要这样实现
            return cat0(nodes);
        } else if (dim == 1) {
            return cat1(nodes);
        }
        return nullptr;
    }

    std::vector<Node *> Node::split0() { // 注意这个函数只用在反向传播中，不需要edge
        std::vector<Matrix *> m_vec = g_backend_ops->split0(this->get_weight());
        std::vector<Node *> res;
        res.reserve(m_vec.size());
        for (uint i = 0; i < m_vec.size(); ++ i) {
            Matrix *m = m_vec[i];
            Node *n = allocNode(m);
            if (is_require_grad()) {
                n->require_grad();
            }
            res.push_back(n);
        }
        return res;
    }

    std::vector<Node *> Node::split1(uint step) {
        Shape shape = this->get_weight()->getShape();
        uint rowCnt = shape.rowCnt;
        assert(step > 0 && rowCnt % step == 0);
        std::vector<Matrix *> m_vec = g_backend_ops->split1(this->get_weight(), step);
        std::vector<Node *> res;
        res.reserve(m_vec.size());
        for (uint i = 0; i < m_vec.size(); ++ i) {
            Matrix *m = m_vec[i];
            Node *n = allocNode(m);
            if (is_require_grad()) {
                n->require_grad();
            }
            n->edges.push_back(SplitEdge1::create(this, i, step));
            res.push_back(n);
        }
        return res;
    }

    std::vector<Node *> Node::split(uint dim, uint step) {
        assert(dim == 0 || dim == 1);
        if (dim == 0) { // 将错就错，dim == 0 时我们切割行，cat也要这样实现
            assert(step == 1);
            return split0();
            
        } else if (dim == 1) {
            return split1(step);
        }
        return {};
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

    void CrossEntropyEdge::backward(Matrix *) {
        assert(node->is_require_grad());
        g_backend_ops->CrossEntropyEdgeBackward(
            node->get_weight(),
            node->get_grad(),
            labels,
            maxs,
            sums
        );
    }

    void CrossEntropyMaskEdge::backward(Matrix *) {
        assert(node->is_require_grad());
        g_backend_ops->CrossEntropyMaskEdgeBackward(
            node->get_weight(),
            node->get_grad(),
            labels,
            info,
            mask
        );
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