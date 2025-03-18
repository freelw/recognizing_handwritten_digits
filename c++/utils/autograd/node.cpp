#include "node.h"
#include <assert.h>
#include <iostream>
#include <string.h>
#include "stats/stats.h"

namespace autograd {

    Matrix *CrossEntropyLoss(Matrix *input, const std::vector<uint> &labels, std::vector<CrosEntropyInfo> &info) { 
        assert(input->getShape().colCnt == labels.size());
        assert(info.size() == 0);
        Matrix *loss = allocTmpMatrix(Shape(1,1));
        DATATYPE loss_value = 0;
        info.resize(input->getShape().colCnt);

        #pragma omp parallel for reduction(+:loss_value)
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
            CrosEntropyInfo &p = info[j];
            p.sum = sum;
            p.max = max;
            loss_value += -(zt - max - log(sum));
        }
        (*loss)[0][0] = loss_value/labels.size();
        return loss;
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
        std::vector<CrosEntropyInfo> info;
        auto *node = allocNode(::autograd::CrossEntropyLoss(w, labels, info));
        assert(info.size() == w->getShape().colCnt);
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(CrossEntropyEdge::create(this, labels, info));
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
        Shape shape = nodes[0]->get_weight()->getShape();
        for (uint i = 0; i < nodes.size(); ++ i) {
            nodes[i]->checkShape(shape);
        }
        Matrix *m = allocTmpMatrix(Shape(shape.rowCnt, shape.colCnt * nodes.size()));
        Node *node = allocNode(m);
        
        auto m_buffer = m->getData();
        auto m_shape = m->getShape();
        
        for (uint i = 0; i < nodes.size(); ++ i) {
            auto node_i_buffer = nodes[i]->get_weight()->getData();
            
            uint k = 0;

            for (; k < shape.rowCnt - 7; k += 8) {

                DATATYPE *m_buffer_0 = m_buffer + k*m_shape.colCnt+i*shape.colCnt;
                DATATYPE *m_buffer_1 = m_buffer + (k+1)*m_shape.colCnt+i*shape.colCnt;
                DATATYPE *m_buffer_2 = m_buffer + (k+2)*m_shape.colCnt+i*shape.colCnt;
                DATATYPE *m_buffer_3 = m_buffer + (k+3)*m_shape.colCnt+i*shape.colCnt;
                DATATYPE *m_buffer_4 = m_buffer + (k+4)*m_shape.colCnt+i*shape.colCnt;
                DATATYPE *m_buffer_5 = m_buffer + (k+5)*m_shape.colCnt+i*shape.colCnt;
                DATATYPE *m_buffer_6 = m_buffer + (k+6)*m_shape.colCnt+i*shape.colCnt;
                DATATYPE *m_buffer_7 = m_buffer + (k+7)*m_shape.colCnt+i*shape.colCnt;

                DATATYPE *node_i_buffer_0 = node_i_buffer + k*shape.colCnt;
                DATATYPE *node_i_buffer_1 = node_i_buffer + (k+1)*shape.colCnt;
                DATATYPE *node_i_buffer_2 = node_i_buffer + (k+2)*shape.colCnt;
                DATATYPE *node_i_buffer_3 = node_i_buffer + (k+3)*shape.colCnt;
                DATATYPE *node_i_buffer_4 = node_i_buffer + (k+4)*shape.colCnt;
                DATATYPE *node_i_buffer_5 = node_i_buffer + (k+5)*shape.colCnt;
                DATATYPE *node_i_buffer_6 = node_i_buffer + (k+6)*shape.colCnt;
                DATATYPE *node_i_buffer_7 = node_i_buffer + (k+7)*shape.colCnt;

                DATATYPE *m_buffers[8] = {
                    m_buffer_0, m_buffer_1, m_buffer_2, m_buffer_3,
                    m_buffer_4, m_buffer_5, m_buffer_6, m_buffer_7
                };

                DATATYPE *node_i_buffers[8] = {
                    node_i_buffer_0, node_i_buffer_1, node_i_buffer_2, node_i_buffer_3,
                    node_i_buffer_4, node_i_buffer_5, node_i_buffer_6, node_i_buffer_7
                };

                #pragma omp parallel for num_threads(8)
                for (uint j = 0; j < 8; ++ j) {
                    memcpy(m_buffers[j], node_i_buffers[j], shape.colCnt * sizeof(DATATYPE));
                }
            }
            for (; k < shape.rowCnt; ++ k) {
                for (uint j = 0; j < shape.colCnt; ++ j) {
                    m_buffer[k*m_shape.colCnt+i*shape.colCnt+j] = node_i_buffer[k*shape.colCnt+j];
                    // (*m)[k][i*shape.colCnt+j] = (*nodes[i]->get_weight())[k][j];
                }
            }
        }
        for (uint i = 0; i < nodes.size(); ++ i) {
            if (nodes[i]->is_require_grad()) {
                node->require_grad();
                node->edges.push_back(CatEdge::create(nodes[i], i*shape.colCnt));
            }
        }
        return node;
    }

    Node *cat1(const std::vector<Node *> &nodes) {
        return nullptr;
    }

    Node *cat(const std::vector<Node *> &nodes, uint dim) {
        assert(dim == 0 || dim == 1);
        assert(nodes.size() > 0);
        if (dim == 0) {
            return cat0(nodes);
        } else if (dim == 1) {
            return cat1(nodes);
        }
        return nullptr;
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

    TmpNodesStats tmpNodesStats() {
        TmpNodesStats stats;
        uint size = nodes.size();
        uint bytes = size * sizeof(Node);
        stats.size = size;
        stats.bytes = bytes;
        return stats;
    }

    TmpEdgesStats tmpEdgesStats() {
        TmpEdgesStats stats;
        uint size = edges.size();
        uint bytes = size * sizeof(Edge);
        stats.size = size;
        stats.bytes = bytes;
        return stats;
    }
} // namespace autograd