#include "liner.h"
#include "xavier.h"

namespace autograd {
    Liner::Liner(uint input_num, uint output_num, bool _bias) 
        : bias(_bias) {
        mW = new Matrix(Shape(output_num, input_num));
        init_weight(mW, xavier_init_sigmoid(mW));
        W = new Node(mW, true);
        W->require_grad();
        PW = new Parameters(W);

        if (bias) {
            mb = new Matrix(Shape(output_num, 1));
            mb->zero();
            b = new Node(mb, true);
            b->require_grad();
            Pb = new Parameters(b);
        }
    }

    Liner::~Liner() {
        delete mW;
        delete W;
        delete PW;
        if (bias) {
            delete mb;
            delete b;
            delete Pb;
        }
    }

    Node *Liner::forward(Node *input) {
        auto node = W->at(input);
        return bias ? node->expand_add(b) : node;
    }

    std::vector<Parameters *> Liner::get_parameters() {
        std::vector<Parameters *> res;
        res.push_back(PW);
        if (bias) {
            res.push_back(Pb);
        }
        return res;
    }

    LazyLiner::LazyLiner(uint _output_num, bool _bias) 
        : output_num(_output_num), bias(_bias) {
    }

    LazyLiner::~LazyLiner() {
        if (liner != nullptr) {
            delete liner;
        }
    }

    Node *LazyLiner::forward(Node *input) {
        if (liner == nullptr) {
            liner = new Liner(input->get_weight()->getShape().rowCnt, output_num, bias);
        }
        return liner->forward(input);
    }

    std::vector<Parameters *> LazyLiner::get_parameters() {
        assert(liner != nullptr); // adam 必须在每一轮结束获取一次参数
        return liner->get_parameters();
    }
    
} // namespace autograd
