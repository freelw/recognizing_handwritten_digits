#include "rnnlm.h"
#include "lmcommon/common.h"

namespace autograd {
    LSTM::LSTM(uint input_num, uint _hidden_num, DATATYPE sigma) : hidden_num(_hidden_num) {
        mWxi = new Matrix(Shape(hidden_num, input_num));
        mWhi = new Matrix(Shape(hidden_num, hidden_num));
        mBi = new Matrix(Shape(hidden_num, 1));
        mWxf = new Matrix(Shape(hidden_num, input_num));
        mWhf = new Matrix(Shape(hidden_num, hidden_num));
        mBf = new Matrix(Shape(hidden_num, 1));
        mWxo = new Matrix(Shape(hidden_num, input_num));
        mWho = new Matrix(Shape(hidden_num, hidden_num));
        mBo = new Matrix(Shape(hidden_num, 1));
        mWxc = new Matrix(Shape(hidden_num, input_num));
        mWhc = new Matrix(Shape(hidden_num, hidden_num));
        mBc = new Matrix(Shape(hidden_num, 1));
        init_weight(mWxi, sigma);
        init_weight(mWhi, sigma);
        init_weight(mBi, sigma);
        init_weight(mWxf, sigma);
        init_weight(mWhf, sigma);
        init_weight(mBf, sigma);
        init_weight(mWxo, sigma);
        init_weight(mWho, sigma);
        init_weight(mBo, sigma);
        init_weight(mWxc, sigma);
        init_weight(mWhc, sigma);
        init_weight(mBc, sigma);
        Wxi = new Node(mWxi, true);
        Whi = new Node(mWhi, true);
        Bi = new Node(mBi, true);
        Wxf = new Node(mWxf, true);
        Whf = new Node(mWhf, true);
        Bf = new Node(mBf, true);
        Wxo = new Node(mWxo, true);
        Who = new Node(mWho, true);
        Bo = new Node(mBo, true);
        Wxc = new Node(mWxc, true);
        Whc = new Node(mWhc, true);
        Bc = new Node(mBc, true);
        Wxi->require_grad();
        Whi->require_grad();
        Bi->require_grad();
        Wxf->require_grad();
        Whf->require_grad();
        Bf->require_grad();
        Wxo->require_grad();
        Who->require_grad();
        Bo->require_grad();
        Wxc->require_grad();
        Whc->require_grad();
        Bc->require_grad();
        PWxi = new Parameters(Wxi);
        PWhi = new Parameters(Whi);
        PBi = new Parameters(Bi);
        PWxf = new Parameters(Wxf);
        PWhf = new Parameters(Whf);
        PBf = new Parameters(Bf);
        PWxo = new Parameters(Wxo);
        PWho = new Parameters(Who);
        PBo = new Parameters(Bo);
        PWxc = new Parameters(Wxc);
        PWhc = new Parameters(Whc);
        PBc = new Parameters(Bc);
    }

    LSTM::~LSTM() {
        delete mWxi;
        delete mWhi;
        delete mBi;
        delete mWxf;
        delete mWhf;
        delete mBf;
        delete mWxo;
        delete mWho;
        delete mBo;
        delete mWxc;
        delete mWhc;
        delete mBc;
        delete Wxi;
        delete Whi;
        delete Bi;
        delete Wxf;
        delete Whf;
        delete Bf;
        delete Wxo;
        delete Who;
        delete Bo;
        delete Wxc;
        delete Whc;
        delete Bc;
        delete PWxi;
        delete PWhi;
        delete PBi;
        delete PWxf;
        delete PWhf;
        delete PBf;
        delete PWxo;
        delete PWho;
        delete PBo;
        delete PWxc;
        delete PWhc;
        delete PBc;
    }

    std::vector<std::pair<Node *, Node*>> LSTM::forward(const std::vector<Node *> &inputs, Node *prev_hidden, Node *prev_cell) {
        assert(inputs.size() > 0);
        uint batch_size = inputs[0]->get_weight()->getShape().colCnt;
        Node *hidden = nullptr;
        Node *cell = nullptr;
        if (prev_hidden == nullptr) {
            hidden = allocNode(allocTmpMatrix(Shape(hidden_num, batch_size)));
        } else {
            hidden = prev_hidden;
        }
        if (prev_cell == nullptr) {
            cell = allocNode(allocTmpMatrix(Shape(hidden_num, batch_size)));
        } else {
            cell = prev_cell;
        }
        std::vector<std::pair<Node *, Node*>> res;
        for (auto input : inputs) {
            Node *i = (*(Wxi->at(input)) + Whi->at(hidden))->expand_add(Bi);
            Node *f = (*(Wxf->at(input)) + Whf->at(hidden))->expand_add(Bf);
            Node *o = (*(Wxo->at(input)) + Who->at(hidden))->expand_add(Bo);
            Node *c = (*(Wxc->at(input)) + Whc->at(hidden))->expand_add(Bc);
            Node *f_sigmoid = f->Sigmoid();
            Node *i_sigmoid = i->Sigmoid();
            Node *o_sigmoid = o->Sigmoid();
            Node *c_tanh = c->Tanh();
            cell = (*(*f_sigmoid * cell) + (*i_sigmoid * c_tanh));
            Node *cell_tanh = cell->Tanh();
            hidden = *o_sigmoid * cell_tanh;
            res.push_back(std::make_pair(hidden, cell));
        }
        return res;
    }

    std::vector<Parameters *> LSTM::get_parameters() {
        return {PWxi, PWhi, PBi, PWxf, PWhf, PBf, PWxo, PWho, PBo, PWxc, PWhc, PBc};
    }

    RnnLM::RnnLM(LSTM *_rnn, uint _vocab_size) : rnn(_rnn), vocab_size(_vocab_size) {
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

    Node *RnnLM::forward(std::vector<Node *> inputs) {
        assert(inputs.size() > 0);
        Shape shape = inputs[0]->get_weight()->getShape();
        std::vector<std::pair<Node *, Node*>> pairs = rnn->forward(inputs, nullptr, nullptr);
        std::vector<Node *> outputs;
        for (auto pair : pairs) {
            Node *hidden = pair.first;
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
        std::vector<std::pair<Node *, Node*>> pairs = rnn->forward(inputs, nullptr, nullptr);
        auto size = pairs.size();
        assert(prefix.length() == size);
        Node *hidden = pairs[size - 1].first;
        Node *cell = pairs[size - 1].second;
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
            pairs = rnn->forward({input}, hidden, cell);
            assert(pairs.size() == 1);
            hidden = pairs[0].first;
            cell = pairs[0].second;
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