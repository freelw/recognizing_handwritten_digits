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
    
} // namespace autograd
