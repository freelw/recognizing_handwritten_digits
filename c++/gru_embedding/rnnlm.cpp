#include "rnnlm.h"
#include "lmcommon/common.h"

namespace autograd {

    Embedding::Embedding(uint _vocab_size, uint _hidden_num) : vocab_size(_vocab_size), hidden_num(_hidden_num) {
        mW = new Matrix(Shape(hidden_num, vocab_size));
        init_weight(mW, 0.02);
        W = new Node(mW, true);
        W->require_grad();
        PW = new Parameters(W);
    }

    Embedding::~Embedding() {
        delete mW;
        delete W;
        delete PW;
    }

    std::vector<Node *> Embedding::forward(const std::vector<Node *> &inputs) {
        std::vector<Node *> res;
        for (auto input : inputs) {
            res.push_back(W->at(input));
        }
        return res;
    }

    std::vector<Parameters *> Embedding::get_parameters() {
        std::vector<Parameters *> res;
        res.push_back(PW);
        return res;
    }

    GRU::GRU(uint input_num, uint _hidden_num, DATATYPE sigma) : hidden_num(_hidden_num) {
        mWxr = new Matrix(Shape(hidden_num, input_num));
        mWhr = new Matrix(Shape(hidden_num, hidden_num));
        mBr = new Matrix(Shape(hidden_num, 1));
        mWxz = new Matrix(Shape(hidden_num, input_num));
        mWhz = new Matrix(Shape(hidden_num, hidden_num));
        mBz = new Matrix(Shape(hidden_num, 1));
        mWxh = new Matrix(Shape(hidden_num, input_num));
        mWhh = new Matrix(Shape(hidden_num, hidden_num));
        mBh = new Matrix(Shape(hidden_num, 1));
        init_weight(mWxr, sigma);
        init_weight(mWhr, sigma);
        init_weight(mBr, sigma);
        init_weight(mWxz, sigma);
        init_weight(mWhz, sigma);
        init_weight(mBz, sigma);
        init_weight(mWxh, sigma);
        init_weight(mWhh, sigma);
        init_weight(mBh, sigma);
        Wxr = new Node(mWxr, true);
        Whr = new Node(mWhr, true);
        Br = new Node(mBr, true);
        Wxz = new Node(mWxz, true);
        Whz = new Node(mWhz, true);
        Bz = new Node(mBz, true);
        Wxh = new Node(mWxh, true);
        Whh = new Node(mWhh, true);
        Bh = new Node(mBh, true);
        Wxr->require_grad();
        Whr->require_grad();
        Br->require_grad();
        Wxz->require_grad();
        Whz->require_grad();
        Bz->require_grad();
        Wxh->require_grad();
        Whh->require_grad();
        Bh->require_grad();
        PWxr = new Parameters(Wxr);
        PWhr = new Parameters(Whr);
        PBr = new Parameters(Br);
        PWxz = new Parameters(Wxz);
        PWhz = new Parameters(Whz);
        PBz = new Parameters(Bz);
        PWxh = new Parameters(Wxh);
        PWhh = new Parameters(Whh);
        PBh = new Parameters(Bh);
    }

    GRU::~GRU() {
        delete mWxr;
        delete mWhr;
        delete mBr;
        delete mWxz;
        delete mWhz;
        delete mBz;
        delete mWxh;
        delete mWhh;
        delete mBh;
        delete Wxr;
        delete Whr;
        delete Br;
        delete Wxz;
        delete Whz;
        delete Bz;
        delete Wxh;
        delete Whh;
        delete Bh;
        delete PWxr;
        delete PWhr;
        delete PBr;
        delete PWxz;
        delete PWhz;
        delete PBz;
        delete PWxh;
        delete PWhh;
        delete PBh;
    }

    std::vector< Node*> GRU::forward(const std::vector<Node *> &inputs, Node *prev_hidden) {
        assert(inputs.size() > 0);
        uint batch_size = inputs[0]->get_weight()->getShape().colCnt;
        Node *hidden = nullptr;
        if (prev_hidden == nullptr) {
            hidden = allocNode(allocTmpMatrix(Shape(hidden_num, batch_size)));
        } else {
            hidden = prev_hidden;
        }
        std::vector<Node *> res;
        
        for (auto input : inputs) {
            Node *r = (*(Wxr->at(input)) + Whr->at(hidden))->expand_add(Br)->Sigmoid();
            Node *z = (*(Wxz->at(input)) + Whz->at(hidden))->expand_add(Bz)->Sigmoid();
            Node *h_tilde = (*(Wxh->at(input)) + Whh->at(*r * hidden))->expand_add(Bh)->Tanh();
            hidden = *(*z * hidden) + *(1 - *z) * h_tilde;
            res.push_back(hidden);
        }
        return res;
    }

    std::vector<Parameters *> GRU::get_parameters() {
        std::vector<Parameters *> res;
        res.push_back(PWxr);
        res.push_back(PWhr);
        res.push_back(PBr);
        res.push_back(PWxz);
        res.push_back(PWhz);
        res.push_back(PBz);
        res.push_back(PWxh);
        res.push_back(PWhh);
        res.push_back(PBh);
        return res;
    }

    RnnLM::RnnLM(GRU *_rnn, Embedding *_embedding, uint _vocab_size) 
        : rnn(_rnn), embedding(_embedding), vocab_size(_vocab_size) {
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

    Node *RnnLM::forward(const std::vector<Node *> &inputs) {
        assert(inputs.size() > 0);
        Shape shape = inputs[0]->get_weight()->getShape();
        std::vector<Node *> embs = embedding->forward(inputs);
        std::vector<Node *> hiddens = rnn->forward(embs, nullptr);
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

    std::vector<uint> RnnLM::predict(const std::vector<uint> &token_ids, uint num_preds) {
        assert(token_ids.size() > 0);
        std::vector<Node *> inputs;
        for (uint i = 0; i < token_ids.size(); i++) {
            Matrix *m = allocTmpMatrix(Shape(vocab_size, 1));
            (*m)[token_ids[i]][0] = 1;
            inputs.push_back(autograd::allocNode(m));
        }
        std::vector<Node *> hiddens = rnn->forward(inputs, nullptr);
        auto size = hiddens.size();
        assert(token_ids.size() == size);
        Node *hidden = hiddens[size - 1];
        assert(hidden->getShape().colCnt == 1);
        std::vector<uint> res;
        for (uint i = 0; i < num_preds; ++ i) {
            auto output = output_layer(hidden);
            std::vector<uint> v_max = output->get_weight()->argMax();
            assert(v_max.size() == 1);
            auto max_index = v_max[0];
            res.push_back(max_index);
            Matrix *m = allocTmpMatrix(Shape(vocab_size, 1));
            (*m)[max_index][0] = 1;
            Node *input = autograd::allocNode(m);
            hiddens = rnn->forward({input}, hidden);
            assert(hiddens.size() == 1);
            hidden = hiddens[0];
        }
        return res;
    }

    std::vector<Parameters *> RnnLM::get_parameters() {
        std::vector<Parameters *> res;
        res.push_back(PW);
        res.push_back(Pb);
        auto embedding_params = embedding->get_parameters();
        res.insert(res.end(), embedding_params.begin(), embedding_params.end());
        auto rnn_params = rnn->get_parameters();
        res.insert(res.end(), rnn_params.begin(), rnn_params.end());
        return res;
    }

} // namespace autograd