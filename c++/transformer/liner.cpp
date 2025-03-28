#include "liner.h"
#include "xavier.h"

namespace autograd {
    Liner::Liner(uint input_num, uint output_num) {
        mW = new Matrix(Shape(output_num, input_num));
        mb = new Matrix(Shape(output_num, 1));
        init_weight(mW, xavier_init_sigmoid(mW));
        mb->zero();
        W = new Node(mW, true);
        b = new Node(mb, true);
        W->require_grad();
        b->require_grad();
        PW = new Parameters(W);
        Pb = new Parameters(b);
    }

    Liner::~Liner() {
        delete mW;
        delete mb;
        delete W;
        delete b;
        delete PW;
        delete Pb;
    }

    Node *Liner::forward(Node *input) {
        return W->at(input)->expand_add(b);
    }

    std::vector<Parameters *> Liner::get_parameters() {
        std::vector<Parameters *> res;
        res.push_back(PW);
        res.push_back(Pb);
        return res;
    }
    
} // namespace autograd
