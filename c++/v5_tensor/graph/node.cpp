#include "node.h"
#include "actions.h"

namespace graph {

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

    Node *Node::transpose(int a, int b) {
        Tensor *l_tensor = this->get_tensor();
        Tensor *res_tensor = l_tensor->transpose(a, b);
        Node *res_node = nullptr;
        if (is_require_grad()) {
            res_node = allocNode(res_tensor, this->get_grad()->transpose(a, b));
            res_node->require_grad();
            res_node->edges.push_back(EmptyEdge::create(this));
        } else {
            res_node = allocNode(res_tensor);
        }
        return res_node;
    }

    Node *Node::reshape(const std::vector<int> &shape) {
        Tensor *l_tensor = this->get_tensor();
        Tensor *res_tensor = l_tensor->reshape(shape);
        bool share_mem = l_tensor->is_shared_with(res_tensor);
        assert(l_tensor->is_contiguous() == share_mem);
        Node *res_node = nullptr;
        if (is_require_grad()) {
            Tensor *grad = this->get_grad();
            Tensor *res_grad = grad->reshape(shape);
            assert(grad->is_contiguous() == share_mem);
            assert(grad->is_shared_with(res_grad) == share_mem);
            res_node = allocNode(res_tensor, res_grad);
            res_node->require_grad();
            if (share_mem) {
                res_node->edges.push_back(EmptyEdge::create(this));
            } else {
                res_node->edges.push_back(ReshapeEdge::create(this));
            }   
        } else {
            res_node = allocNode(res_tensor);
        }
        return res_node;
    }

    Node *Node::sequence_mask(Tensor *mask, float value) {
        Tensor *res_tensor = this->get_tensor()->sequence_mask(mask, value);
        Node *res_node = nullptr;
        if (is_require_grad()) {
            res_node = allocNode(res_tensor, this->get_grad());
            res_node->edges.push_back(EmptyEdge::create(this));
        } else {
            res_node = allocNode(res_tensor);
        }
        return res_node;
    }

    Node *Node::softmax() {
        assert(this->get_tensor()->get_dim() == 3);
        auto shape = this->get_tensor()->get_shape();
        auto res_node = allocNode(
            this->get_tensor()->softmax()
        );
        if (is_require_grad()) {
            res_node->require_grad();
            res_node->edges.push_back(SoftmaxEdge::create(this, res_node->get_tensor()));
        }
        return res_node;
    }

    Node *Node::masked_softmax(Tensor *valid_len) {
        assert(this->get_tensor()->get_dim() == 3);
        if (valid_len == nullptr) {
            return this->softmax();
        } else {
            auto shape = this->get_tensor()->get_shape();
            Tensor *mask = valid_len->get_dim() == 1 ?  
                valid_len->repeat_interleave(shape[1]) : valid_len->reshape({-1});
            return this->reshape({-1, shape[2]})
                ->sequence_mask(mask, -1e6f)
                ->reshape(shape)
                ->softmax();
        }
    }

    Node *Node::expand_add(Node *rhs) {
        Tensor *res_tensor = allocTensor(t->get_shape(), "expand_add");
        Tensor *r_tensor = rhs->get_tensor();
        gCreateAction(
            new ExpandAddAction(
                this->get_tensor(),
                rhs->get_tensor(),
                res_tensor
            )
        );
        Node *res_node = allocNode(res_tensor);
        if (is_require_grad() || rhs->is_require_grad()) {
            res_node->require_grad();
            if (is_require_grad()) {
                res_node->edges.push_back(AddEdge::create(this));
            }
            if (rhs->is_require_grad()) {
                res_node->edges.push_back(ExpandAddEdge::create(rhs));
            }
        }
        return res_node;
    }

    Node *Node::at(Node *rhs) {
        Tensor *r_tensor = rhs->get_tensor();
        Tensor *l_tensor = this->get_tensor();
        assert(l_tensor->get_dim() == 2);
        assert(r_tensor->get_dim() == 2);
        assert(l_tensor->get_shape()[1] == r_tensor->get_shape()[0]);
        Tensor *res_tensor = allocTensor({l_tensor->get_shape()[0], r_tensor->get_shape()[1]}, "res_at");
        gCreateAction(
            new AtAction(
                this->get_tensor(),
                rhs->get_tensor(),
                res_tensor
            )
        );
        Node *res_node = allocNode(res_tensor);
        if (is_require_grad() || rhs->is_require_grad()) {
            res_node->require_grad();
            if (is_require_grad()) {
                res_node->edges.push_back(MatMulLEdge::create(this, rhs));
            }
            if (rhs->is_require_grad()) {
                res_node->edges.push_back(MatMulREdge::create(rhs, this));
            }
        }
        return res_node;
    }

    Node *Node::bmm(Node *rhs) {
        Node *l_node = this;
        Node *r_node = rhs;

        if (!l_node->get_tensor()->is_contiguous()) {
            l_node = l_node->reshape(l_node->get_tensor()->get_shape());
        }
        if (!r_node->get_tensor()->is_contiguous()) {
            r_node = r_node->reshape(r_node->get_tensor()->get_shape());
        }

        auto l_tensor = l_node->get_tensor();
        auto r_tensor = r_node->get_tensor();
        assert(l_tensor->get_dim() == 3);
        assert(r_tensor->get_dim() == 3);
        assert(l_tensor->get_shape()[2] == r_tensor->get_shape()[1]);
        
        if (l_node->is_require_grad()) {
            auto l_grad = l_node->get_grad();
            assert(l_tensor->get_shape() == l_grad->get_shape());
            assert(l_grad->get_dim() == 3);
            assert(l_tensor->is_contiguous() == l_grad->is_contiguous());
        }
        
        if (r_node->is_require_grad()) {
            auto r_grad = r_node->get_grad();
            assert(r_tensor->get_shape() == r_grad->get_shape());
            assert(r_grad->get_dim() == 3);
            assert(r_tensor->is_contiguous() == r_grad->is_contiguous());
        }
        std::vector<Node *> l_split_2d_nodes;
        std::vector<Node *> r_split_2d_nodes;

        l_node->split_3d(l_split_2d_nodes);
        r_node->split_3d(r_split_2d_nodes);

        assert(l_split_2d_nodes.size() == r_split_2d_nodes.size());


        assert(false);
        return nullptr;
    }

    void Node::split_3d(std::vector<Node *> &res_nodes) {
        assert(this->get_tensor()->get_dim() == 3);
        if (this->is_require_grad()) {
            assert(this->get_grad()->get_dim() == 3);
        }
        auto shape = this->get_tensor()->get_shape();
        res_nodes.clear();
        res_nodes.reserve(shape[0]);
        int offset = 0;
        int block = shape[1] * shape[2];
        for (int i = 0; i < shape[0]; ++i) {
            Node *node = nullptr;
            std::vector<int> new_strides;
            new_strides.resize(2);
            new_strides[0] = shape[2];
            new_strides[1] = 1;
            Tensor *new_tensor = allocTensorView(
                this->get_tensor(),
                {shape[1], shape[2]},
                new_strides,
                this->get_tensor()->get_name() + "_split_" + std::to_string(i)
            );
            if (this->is_require_grad()) {
                Tensor *new_grad = allocTensorView(
                    this->get_grad(),
                    {shape[1], shape[2]},
                    new_strides,
                    this->get_grad()->get_name() + "_split_" + std::to_string(i)
                );
                node = allocNode(new_tensor, new_grad);
                node->edges.push_back(EmptyEdge::create(this));
            } else {
                node = allocNode(new_tensor);
            }
            res_nodes.push_back(node);
        }
    }

    Node *Node::relu() {
        Tensor *l_tensor = this->get_tensor();
        Tensor *res_tensor = allocTensor(l_tensor->get_shape(), "relu_res");
        gCreateAction(
            new ReluAction(
                l_tensor,
                res_tensor
            )
        );
        Node *res_node = allocNode(res_tensor);
        if (is_require_grad()) {
            res_node->require_grad();
            res_node->edges.push_back(ReluEdge::create(this));
        }
        return res_node;
    }

    Node *Node::CrossEntropy(Tensor *labels) {
        assert(labels->get_dim() == 1);
        assert(
            labels->get_dtype() == INT32 
        );
        assert(labels->get_shape()[0] == this->get_tensor()->get_shape()[0]);
        Tensor *tensor_maxs = allocTensor(labels->get_shape(), "maxs");
        Tensor *tensor_sums = allocTensor(labels->get_shape(), "sums");
        Tensor *ce_res = allocTensor({1}, "cross_entropy");

        gCreateAction(
            new CrossEntropyAction(
                this->get_tensor(),
                labels,
                tensor_maxs,
                tensor_sums,
                ce_res
            )
        );
        Node *res_node = allocNode(ce_res);
        if (is_require_grad() ) {
            res_node->require_grad();
            res_node->edges.push_back(CrossEntropyEdge::create(this, labels, tensor_maxs, tensor_sums));
        }
        return res_node;
    }

    void Node::init_weight_gauss(float sigma, float mean) {
        gCreateAction(
            new InitWeightAction(
                this->get_tensor(),
                "gauss",
                sigma,
                mean
            )
        );
    }

    void Node::init_weight_uniform(float sigma) {
        gCreateAction(
            new InitWeightAction(
                this->get_tensor(),
                "uniform",
                sigma,
                0
            )
        );
    }

    void Node::init_weight_for_dbg(float scale) {
        gCreateAction(
            new InitWeightAction(
                this->get_tensor(),
                "dbg",
                scale,
                0
            )
        );
    }

    void CrossEntropyEdge::backward(Tensor *) {
        gCreateAction(
            new CrossEntropyBackwardAction(
                node->get_tensor(),
                labels,
                maxs,
                sums,
                node->get_grad()
            )
        );
    }

    void SoftmaxEdge::backward(Tensor *grad) {
        gCreateAction(
            new SoftmaxBackwardAction(
                node->get_grad(),
                softmax_res,
                grad
            )
        );
    }

    std::vector<Edge *> edges;
    std::vector<Node *> nodes;

    Node *allocNode(Tensor *t) {
        Node *node = new Node(t);
        nodes.push_back(node);
        return node;
    }

    Node *allocNode(Tensor *t, Tensor *grad) {
        Node *node = new Node(t, grad);
        nodes.push_back(node);
        return node;
    }

    void validateAllNodes() {
        for (Node *node : nodes) {
            if (node->is_require_grad()) {
                bool grad_contiguous = node->get_grad()->is_contiguous();
                bool tensor_contiguous = node->get_tensor()->is_contiguous();
                assert(grad_contiguous == tensor_contiguous);
                auto grad_shape = node->get_grad()->get_shape();
                auto tensor_shape = node->get_tensor()->get_shape();
                assert(grad_shape == tensor_shape);
            }
        }
    }

    void gAddEdge(Edge *edge) {
        edges.push_back(edge);
    }

    void freeAllNodes() {
        for (Node *node : nodes) {
            delete node;
        }
        nodes.clear();
    }

    void freeAllEdges() {
        for (Edge *edge : edges) {
            delete edge;
        }
        edges.clear();
    }

} // namespace graph