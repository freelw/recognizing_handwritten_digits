#include "rnnlm.h"

namespace autograd {
    Rnn::Rnn(uint input_num, uint _hidden_num, DATATYPE sigma) : hidden_num(_hidden_num) {
        mWxh = new Matrix(Shape(hidden_num, input_num));
        mWhh = new Matrix(Shape(hidden_num, hidden_num));
        mbh = new Matrix(Shape(hidden_num, 1));
        
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

    std::vector<Node *> Rnn::forward(const std::vector<Node *> &inputs, Node *prev_hidden) {
        assert(inputs.size() > 0);
        uint batch_size = inputs[0]->get_weight()->getShape().colCnt;
        Node *hidden = nullptr;
        if (prev_hidden == nullptr) {
            hidden = allocNode(allocTmpMatrix(Shape(mWhh->getShape().rowCnt, batch_size)));
        } else {
            hidden = prev_hidden;
        }
        std::vector<Node *> res;
        for (auto input : inputs) {
            hidden = (*(Wxh->at(input)) + Whh->at(hidden))->expand_add(bh)->Tanh();
            res.push_back(hidden);
        }
        return res;
    }

    std::vector<Parameters *> Rnn::get_parameters() {
        return {PWxh, PWhh, Pbh};
    }

    RnnLM::RnnLM(Rnn *_rnn, uint vocab_size) : rnn(_rnn) {
        mW = new Matrix(Shape(vocab_size, rnn->get_hidden_num()));
        mb = new Matrix(Shape(vocab_size, 1));
        init_weight(mW, 0.02);
        init_weight(mb, 0.02);
        W = new Node(mW, true);
        b = new Node(mb, true);
        W->require_grad();
        b->require_grad();
        PW = new Parameters(W);
        Pb = new Parameters(b);
    }

    RnnLM::~RnnLM() {
        delete mW;
        delete mb;
        delete W;
        delete b;
        delete PW;
        delete Pb;
    }

    Node * RnnLM::forward(std::vector<Node *> inputs) {
        assert(inputs.size() > 0);
        Shape shape = inputs[0]->get_weight()->getShape();
        std::vector<Node *> hiddens = rnn->forward(inputs, nullptr);
        std::vector<Node *> outputs;
        for (auto hidden : hiddens) {
            auto output = W->at(hidden)->expand_add(b);
            outputs.push_back(output);
        }
        Node *res = cat(outputs);
        assert(res->get_weight()->getShape().rowCnt == shape.rowCnt);
        assert(res->get_weight()->getShape().colCnt == shape.colCnt*outputs.size());
        return res;
    }

    Node * RnnLM::predict(const std::string &prefix, uint max_len) {
        return nullptr;
    }

    std::vector<Parameters *> RnnLM::get_parameters() {
        std::vector<Parameters *> res;
        res.push_back(PW);
        res.push_back(Pb);
        auto rnn_params = rnn->get_parameters();
        res.insert(res.end(), rnn_params.begin(), rnn_params.end());
        return res;
    }

} // namespace autograd