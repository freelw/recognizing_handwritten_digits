#include "rnnlm.h"
#include "lmcommon/common.h"

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

    RnnLM::RnnLM(Rnn *_rnn, uint _vocab_size) : rnn(_rnn), vocab_size(_vocab_size) {
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
            outputs.push_back(output_layer(hidden));
        }
        Node *res = cat(outputs);
        assert(res->get_weight()->getShape().rowCnt == shape.rowCnt);
        assert(res->get_weight()->getShape().colCnt == shape.colCnt*outputs.size());
        return res;
    }

    Node *RnnLM::output_layer(Node *hidden) {
        return W->at(hidden)->expand_add(b);
    }

    std::string RnnLM::predict(const std::string &prefix, uint num_preds) {
        assert(prefix.length() > 0);
        std::vector<Node *> inputs;
        for (uint i = 0; i < prefix.size(); i++) {
            Matrix *m = allocTmpMatrix(Shape(vocab_size, 1));
            (*m)[to_index(prefix[i])][0] = 1;
            inputs.push_back(autograd::allocNode(m));
        }
        std::vector<Node *> hiddens = rnn->forward(inputs, nullptr);
        auto size = hiddens.size();
        assert(prefix.length() == size);
        Node *hidden = hiddens[size - 1];
        assert(hidden->getShape().colCnt == 1);
        std::string ret = "";
        for (uint i = 0; i < num_preds; ++ i) {
            auto output = output_layer(hidden);
            std::vector<uint> v_max = output->get_weight()->argMax();
            assert(v_max.size() == 1);
            auto max_index = v_max[0];
            ret += to_char(max_index);
            Matrix *m = allocTmpMatrix(Shape(vocab_size, 1));
            (*m)[max_index][0] = 1;
            Node *input = autograd::allocNode(m);
            hiddens = rnn->forward({input}, hidden);
            assert(hiddens.size() == 1);
            hidden = hiddens[0];
        }
        return ret;
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