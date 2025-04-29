#ifndef GRAPH_NODE_H
#define GRAPH_NODE_H

#include "tensor/tensor.h"
#include <vector>

namespace graph {
    class Edege;
    class Node {
        public:
            Node(Tensor *_t)
                : t(_t)
                ref_cnt(0) {
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
            Node *operator+(Node *rhs);
            Node *operator+=(Node *rhs);
            Node *operator*(Node *rhs);
            Node *expand_add(Node *rhs);
            Node *at(Node *rhs);
            Node *relu();
            Node *transpose_2d();
        private:
            Matrix *t;
            std::vector<Edge *> edges;
            int ref_cnt;
    };

    enum OpType {
        Add,
        Minus,
        ExpandAdd,
        Mul,
        Sub,
        Div,
        MulSV, // single value
        MatMulL,
        MatMulR,
        Tanh,
        Cat0,
        Cat1,
        Split0,
        Split1,
        Sigmoid,
        Relu,
        CrossEntropy,
        CrossEntropyMask,
        Norm,
        Softmax,
        Transpose,
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

    Node *allocNode(Tensor *t);
}
#endif