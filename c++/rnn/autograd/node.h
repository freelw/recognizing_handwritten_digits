#ifndef AUTOGARD_NODE_H
#define AUTOGARD_NODE_H

#include <assert.h>
#include <vector>

#include "matrix/matrix.h"

namespace autograd {
    enum OpType {
        Add,
        Mul,
        Sub,
        Div,
        MatMul,
        Tanh,
        Sigmoid,
        Relu,
        Softmax,
        CrossEntropy,
        LogSoftmax,
        NLLLoss,
        MSELoss,
        Conv2d,
        MaxPool2d,
        AvgPool2d,
        Dropout,
        Embedding,
        RNN,
        LSTM,
        GRU,
        Linear,
        Flatten,
    };
    class Node;
    class Edge {
    public:
        Edge(
            const std::vector<Matrix *> &_params,
            OpType _type,
            Matrix *_t_grad,
            Node *_node) 
            : params(_params), type(_type),
            node(_node) {
                node->inc_ref();
            }
    private:
        std::vector<Matrix *> params;
        OpType type;
        Node *node;
    friend class Node;
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

        Node *operator+(Node *rhs);
        Node *expand_add(Node *rhs);
        Node *at(Node *rhs);
        Node *Relu();
        // Node *operator*(Node &rhs);
        // Node *operator/(Node &rhs);
        // Node *operator-(Node &rhs);
        // Node *operator-();
        // friend Node *operator-(DATATYPE, Node &rhs);
    private:
        Matrix *w;
        Matrix *grad;
        bool requires_grad;
        std::vector<Edge *> edges;
        int ref_cnt;
    };

    extern std::vector<Edge *> edges;
    extern std::vector<Node *> nodes;

    Node *allocNode(Matrix *w);
    Edge *allocEdge(const std::vector<Matrix *> &_params, OpType _type, Matrix *_t_grad);
    void freeAllNodes();
    void freeAllEdges();
} // namespace autograd

#endif