#include "rnnlm.h"

namespace autograd {
    Rnn::Rnn(uint input_num, uint hidden_num) {
        mWxh = new Matrix(Shape(hidden_num, input_num));
        mWhh = new Matrix(Shape(hidden_num, hidden_num));
        mbh = new Matrix(Shape(hidden_num, 1));
        
        auto sigma = 0.02;
        init_weight(mWxh, sigma);
        init_weight(mWhh, sigma);
        init_weight(mbh, sigma);

        Wxh = new Node(mWxh, true);
        Whh = new Node(mWhh, true);
        bh = new Node(mbh, true);

        Wxh->require_grad();
        Whh->require_grad();
        bh->require_grad();

        PWxh = new Parameters(Wxh);
        PWhh = new Parameters(Whh);
        Pbh = new Parameters(bh);
    }

    Rnn::~Rnn() {
        delete mWxh;
        delete mWhh;
        delete mbh;
        
        delete Wxh;
        delete Whh;
        delete bh;
        
        delete PWxh;
        delete PWhh;
        delete Pbh;
    }

    Node *Rnn::forward(const std::vector<Node *> &input, Node *prev_hidden) {
        // Node *hidden = Wxh->at(input[0])->expand_add(Whh->at(prev_hidden))->expand_add(bh)->Tanh();
        // return hidden;

        return nullptr;
    }
} // namespace autograd