#ifndef GRAPH_NODE_H
#define GRAPH_NODE_H

#include "tensor/tensor.h"
#include <vector>
#include "actions.h"

namespace graph {
    class Edge;
    void gAddEdge(Edge *edge);
    class Node {
        public:
            Node(Tensor *_t)
                : t(_t),
                ref_cnt(0) {
                grad = allocGradTensor(t->get_shape(), t->get_name()+"_grad");
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
            Tensor *get_tensor() const {
                return t;
            }
            Tensor *get_grad() const {
                return grad;
            }
            bool is_require_grad() const {
                return true;
            }
            void backward();
            Node *expand_add(Node *rhs);
            // Node *operator*(Node *rhs);
            Node *at(Node *rhs);
            Node *relu();
            // Node *transpose_2d();
        private:
            Tensor *t;
            Tensor *grad;
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
            virtual void backward(Tensor *grad) = 0;
        protected:
            OpType type;
            Node *node; // node that this edge points to
        friend class Node;
    };

    class AddEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new AddEdge(_node);
                graph::gAddEdge(edge);
                return edge;
            }
            AddEdge(Node *_node)
                : Edge(Add, _node) {}
            virtual ~AddEdge() {}
            void backward(Tensor *grad) override {
                gCreateAction(
                    new AddEqAction(node->get_grad(), grad)
                );
            }
    };

    class ExpandAddEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new ExpandAddEdge(_node);
                graph::gAddEdge(edge);
                return edge;
            }
            ExpandAddEdge(Node *_node)
                : Edge(ExpandAdd, _node) {}
            virtual ~ExpandAddEdge() {}
            void backward(Tensor *grad) override {
                assert(grad->get_shape().size() == 2);
                std::vector<int> shape = {grad->get_shape()[1]};
                Tensor *tmp = allocGradTensor(shape, "sum_tmp"); // 行向量
                gCreateAction(
                    new SumAction(
                        grad,
                        tmp,
                        0
                    )
                );
                gCreateAction(
                    new AddEqAction(
                        node->get_grad(),
                        tmp
                    )
                );
            }
    };

    class MatMulLEdge : public Edge {
        public:
            static Edge* create(Node *_node, Node *_rhs) {
                Edge *edge = new MatMulLEdge(_node, _rhs);
                gAddEdge(edge);
                return edge;
            }
            MatMulLEdge(Node *_node, Node *_rhs)
                : Edge(MatMulL, _node), rhs(_rhs) {}
            virtual ~MatMulLEdge() {}
            void backward(Tensor *grad) override {
                Tensor *r_tensor = rhs->get_tensor();
                Tensor *l_tensor = node->get_tensor();
                Tensor *r_transpose_view = allocTensorView(
                    r_tensor,
                    {r_tensor->get_shape()[1], r_tensor->get_shape()[0]},
                    r_tensor->get_name() + "_transpose"
                );

                gCreateAction(
                    new AtAction(
                        grad,
                        r_transpose_view,
                        node->get_grad()
                    )
                );

                // gCreateAction(
                //     new MatMulLEqAction(
                //         node->get_grad(),
                //         grad,
                //         r_tensor
                //     )
                // );
            }
        private:
            Node *rhs;
    };

    Node *allocNode(Tensor *t);
    void freeAllNodes();
    void freeAllEdges();
}
#endif