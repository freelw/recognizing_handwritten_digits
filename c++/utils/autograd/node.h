#ifndef AUTOGARD_NODE_H
#define AUTOGARD_NODE_H

#include <assert.h>
#include <vector>

#include "matrix/matrix.h"

namespace autograd {
    class Edge;
    class Node;
    extern std::vector<Edge *> edges;
    extern std::vector<Node *> nodes;

    enum OpType {
        Add,
        Minus,
        ExpandAdd,
        Mul,
        Sub,
        Div,
        MatMulL,
        MatMulR,
        Tanh,
        Cat,
        Sigmoid,
        Relu,
        CrossEntropy,
    };
    class Node {
        public:
            Node(Matrix *_w, bool magic = false)
                : w(_w), grad(nullptr), requires_grad(false),
                ref_cnt(0) {
                assert(magic);
            }
            void require_grad() {
                requires_grad = true;
                grad = allocTmpMatrix(w->getShape());
            }
            bool is_require_grad() const {
                return requires_grad;
            }
            void inc_ref() {
                ref_cnt++;
            }
            void dec_ref() {
                ref_cnt--;
            }
            int get_ref() const {
                return ref_cnt;
            }

            void backward();
            Matrix *get_weight() {
                return w;
            }
            Matrix *get_grad() {
                return grad;
            }

            Shape getShape() {
                return w->getShape();
            }

            void zero_grad() {
                grad = allocTmpMatrix(w->getShape());                
            }

            void checkShape(const Shape &shape) {
                w->checkShape(shape);
            }

            Node *operator+(Node *rhs);
            Node *operator*(Node *rhs);
            Node *expand_add(Node *rhs);
            Node *at(Node *rhs);
            Node *Relu();
            Node *CrossEntropy(const std::vector<uint> &labels);
            Node *Tanh();
            Node *Sigmoid();
            // Node *operator*(Node &rhs);
            // Node *operator/(Node &rhs);
            // Node *operator-(Node &rhs);
            // Node *operator-();
            friend Node *operator-(DATATYPE, Node &rhs);
            friend Node *cat(const std::vector<Node *> &nodes);
        private:
            Matrix *w;
            Matrix *grad;
            bool requires_grad;
            std::vector<Edge *> edges;
            int ref_cnt;
    };

    class Edge {
        public:
            Edge(
                OpType _type,
                Node *_node) 
                : type(_type),
                node(_node) {
                    node->inc_ref();
                }
            virtual ~Edge() {};
            virtual void backward(Matrix *grad) = 0;
        protected:
            OpType type;
            Node *node; // node that this edge points to
        friend class Node;
    };

    class AddEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new AddEdge(_node);
                edges.push_back(edge);
                return edge;
            }
            AddEdge(Node *_node)
                : Edge(Add, _node) {}
            virtual ~AddEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() += *grad;
            }
    };

    class MinusEdge: public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new MinusEdge(_node);
                edges.push_back(edge);
                return edge;
            }
            MinusEdge(Node *_node)
                : Edge(Minus, _node) {}
            virtual ~MinusEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() -= *grad;
            }
    };

    class MulEdge : public Edge {
        public:
            static Edge* create(Node *_node, Matrix *_param) {
                Edge *edge = new MulEdge(_node, _param);
                edges.push_back(edge);
                return edge;
            }
            MulEdge(Node *_node, Matrix *_param)
                : Edge(Mul, _node), param(_param) {}
            virtual ~MulEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() += *(*grad * *(param));
            }
        private:
            Matrix *param;
    };

    class ExpandAddEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new ExpandAddEdge(_node);
                edges.push_back(edge);
                return edge;
            }
            ExpandAddEdge(Node *_node)
                : Edge(ExpandAdd, _node) {}
            virtual ~ExpandAddEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() += *(grad->sum(1));
            }
    };

    class MatMulLEdge : public Edge {
        public:
            static Edge* create(Node *_node, Matrix *_param) {
                Edge *edge = new MatMulLEdge(_node, _param);
                edges.push_back(edge);
                return edge;
            }
            MatMulLEdge(Node *_node, Matrix *_param)
                : Edge(MatMulL, _node), param(_param) {}
            virtual ~MatMulLEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                // *node->get_grad() is grad of W
                *node->get_grad() += *(grad->at(*(param->transpose())));
            }
        private:
            Matrix *param; // Input Vector
    };

    class MatMulREdge : public Edge {
        public:
            static Edge* create(Node *_node, Matrix *_param) {
                Edge *edge = new MatMulREdge(_node, _param);
                edges.push_back(edge);
                return edge;
            }
            MatMulREdge(Node *_node, Matrix *_param)
                : Edge(MatMulR, _node), param(_param) {}
            virtual ~MatMulREdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                // *node->get_grad() is grad of Input
                *node->get_grad() += *(param->transpose()->at(*grad));
            }
        private:
            Matrix *param; // W
    };

    class ReluEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new ReluEdge(_node);
                edges.push_back(edge);
                return edge;
            }
            ReluEdge(Node *_node)
                : Edge(Relu, _node) {}
            virtual ~ReluEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() += *(*grad * *(node->get_weight()->Relu_prime()));
            }
    };

    struct CrosEntropyInfo {
        DATATYPE sum, max;
    };

    class CrossEntropyEdge : public Edge {
        public:
            static Edge* create(Node *_node, const std::vector<uint> &_labels, const std::vector<CrosEntropyInfo> &_info) { 
                Edge *edge = new CrossEntropyEdge(_node, _labels, _info);
                edges.push_back(edge);
                return edge;
            }
            CrossEntropyEdge(Node *_node, const std::vector<uint> &_labels, const std::vector<CrosEntropyInfo> &_info)
                : Edge(CrossEntropy, _node), labels(_labels), info(_info) {}
            virtual ~CrossEntropyEdge() {}
            void backward(Matrix *) override {
                assert(node->is_require_grad());
                for (uint i = 0; i < labels.size(); ++ i) {
                    auto target = labels[i];
                    DATATYPE max = info[i].max;
                    DATATYPE sum = info[i].sum;
                    for (uint j = 0; j < node->get_weight()->getShape().rowCnt; ++ j) {
                        if (j == target) {
                            continue;
                        }
                        auto &_grad = (*node->get_grad())[j][i];
                        _grad = std::exp((*node->get_weight())[j][i] - max) / sum / labels.size();
                    }
                    (*node->get_grad())[target][i] = (std::exp((*node->get_weight())[target][i] - max) / sum - 1) / labels.size();
                }
            }
        private:
            std::vector<uint> labels;
            std::vector<CrosEntropyInfo> info;
    };

    class TanhEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new TanhEdge(_node);
                edges.push_back(edge);
                return edge;
            }
            TanhEdge(Node *_node)
                : Edge(Tanh, _node) {}
            virtual ~TanhEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() += *(*grad * *(node->get_weight()->tanh_prime()));
            }
    };

    class SigmoidEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new SigmoidEdge(_node);
                edges.push_back(edge);
                return edge;
            }
            SigmoidEdge(Node *_node)
                : Edge(Sigmoid, _node) {}
            virtual ~SigmoidEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() += *(*grad * *(node->get_weight()->sigmoid_prime()));
            }
    };

    class CatEdge: public Edge {
        public:
            static Edge* create(Node *_node, uint _offset) {
                Edge *edge = new CatEdge(_node, _offset);
                edges.push_back(edge);
                return edge;
            }
            CatEdge(Node *_node, uint _offset)
                : Edge(OpType::Cat, _node), offset(_offset){}
            virtual ~CatEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                Shape shape = node->get_weight()->getShape();
                for (uint i = 0; i < shape.rowCnt; ++ i) {
                    for (uint j = 0; j < shape.colCnt; ++ j) {
                        (*node->get_grad())[i][j] += (*grad)[i][j+offset];
                    }
                }
            }
        private:
            uint offset;
    };

    Node *cat(const std::vector<Node *> &nodes);
    Node *allocNode(Matrix *w);
    void freeAllNodes();
    void freeAllEdges();
} // namespace autograd

#endif