#include "seq2seq.h"

namespace autograd {

    Dropout::Dropout(DATATYPE _dropout) : dropout(_dropout) {
        assert(dropout > 0);
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        gen = std::mt19937(seed);
        dis = std::uniform_real_distribution<>(0, 1);
    }

    std::vector<Node *> Dropout::forward(const std::vector<Node *> &inputs) {
        std::vector<Node *> res;
        res.resize(inputs.size());
        for (uint j = 0; j < inputs.size(); j++) {
            auto &input = inputs[j];
            Matrix *mask = allocTmpMatrix(input->get_weight()->getShape());
            auto buffer = mask->getData();
            for (uint i = 0; i < mask->getShape().size(); i++) {
                buffer[i] = dis(gen) > dropout ? 1 : 0;
            }
            Node *n = allocNode(mask);
            res[j] = *input * n;
        }
        return res;
    }

    Embedding::Embedding(uint _vocab_size, uint _hidden_num) : vocab_size(_vocab_size), hidden_num(_hidden_num) {
        for (uint i = 0; i < vocab_size; i++) {
            Matrix *m = new Matrix(Shape(hidden_num, 1));
            init_weight(m, 0.02);
            mW.push_back(m);
            Node *n = new Node(m, true);
            n->require_grad();
            W.push_back(n);
            PW.push_back(new Parameters(n));
        }
    }

    Embedding::~Embedding() {
        for (auto m : mW) {
            delete m;
        }
        for (auto n : W) {
            delete n;
        }
        for (auto p : PW) {
            delete p;
        }
    }

    std::vector<Node *> Embedding::forward(const std::vector<std::vector<uint>> &inputs) {
        std::vector<Node *> res;
        for (auto &input : inputs) {
            std::vector<Node *> tmp;
            for (auto i : input) {
                tmp.push_back(W[i]);
            }
            res.push_back(cat(tmp));
        }
        return res;
    }

    std::vector<Parameters *> Embedding::get_parameters() {
        std::vector<Parameters *> res;
        for (auto p : PW) {
            res.push_back(p);
        }
        return res;
    }

    Liner::Liner(uint input_num, uint output_num, DATATYPE sigma) {
        mW = new Matrix(Shape(output_num, input_num));
        mb = new Matrix(Shape(output_num, 1));
        init_weight(mW, sigma);
        init_weight(mb, sigma);
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

    GRULayer::GRULayer(
        uint input_num,
        uint _hidden_num,
        DATATYPE sigma) : hidden_num(_hidden_num) {

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

    GRULayer::~GRULayer() {
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

    std::vector<Node *> GRULayer::forward(
        const std::vector<Node *> &inputs,
        Node *hidden) {
        
        assert(inputs.size() > 0);
        uint batch_size = inputs[0]->get_weight()->getShape().colCnt;
        if (hidden == nullptr) {
            hidden = allocNode(allocTmpMatrix(Shape(hidden_num, batch_size)));
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

    std::vector<Parameters *> GRULayer::get_parameters() {
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

    GRU::GRU(
        uint input_num, uint _hidden_num, uint _layer_num,
        DATATYPE sigma, DATATYPE _dropout
    ) : hidden_num(_hidden_num), layer_num(_layer_num), dropout(_dropout), training(true) {

        assert(layer_num > 0);
        layers.push_back(new GRULayer(input_num, hidden_num, sigma));
        for (uint i = 1; i < layer_num; i++) {
            layers.push_back(new GRULayer(hidden_num, hidden_num, sigma));
        }
        if (dropout > 0) {
            dropout_layer = new Dropout(dropout);
        }
    }

    GRU::~GRU() {
        if (dropout > 0 && dropout_layer != nullptr) {
            delete dropout_layer;
        }
        for (auto layer : layers) {
            delete layer;
        }
    }

    std::vector<std::vector<Node*>> GRU::forward(
        const std::vector<Node *> &inputs,
        const std::vector<Node *> &hiddens) {

        assert(inputs.size() > 0);
        assert(hiddens.size() == layer_num);
        std::vector<std::vector<Node*>> res;

        for (uint i = 0; i < layer_num; i++) {
            std::vector<Node *> hidden;
            if (i == 0) {
                hidden = layers[i]->forward(inputs, hiddens[i]);
            } else {
                hidden = layers[i]->forward(res[i - 1], hiddens[i]);
            }
            if (training && dropout > 0 && i < layer_num - 1) {
                hidden = dropout_layer->forward(hidden);
            }
            res.push_back(hidden);
        }
        return res;
    }

    std::vector<Parameters *> GRU::get_parameters() {
        std::vector<Parameters *> res;
        for (auto layer : layers) {
            auto params = layer->get_parameters();
            res.insert(res.end(), params.begin(), params.end());
        }
        return res;
    }

    Seq2SeqEncoder::Seq2SeqEncoder(
        uint _vocab_size,
        uint _embed_size,
        uint _hidden_num, uint _layer_num,
        DATATYPE sigma, DATATYPE _dropout
    ) : vocab_size(_vocab_size),
        embed_size(_embed_size),
        hidden_num(_hidden_num),
        layer_num(_layer_num),
        dropout(_dropout),
        training(true) {
        assert(layer_num > 0);
        embedding = new Embedding(vocab_size, embed_size);
        rnn = new GRU(embed_size, hidden_num, layer_num, sigma, dropout);
    }

    Seq2SeqEncoder::~Seq2SeqEncoder() {
        delete rnn;
        delete embedding;
    }

    std::vector<Node*> Seq2SeqEncoder::forward(
        const std::vector<std::vector<uint>> &token_ids,
        std::vector<Node *> &encoder_states) {
        assert(encoder_states.size() == 0);
        assert(token_ids.size() > 0);
        auto batch_size = token_ids[0].size();
        auto num_steps = token_ids.size();
        std::vector<Node *> inputs = embedding->forward(token_ids);
        std::vector<Node *> input_hiddens;
        for (uint i = 0; i < layer_num; i++) {
            input_hiddens.push_back(nullptr);
        }
        std::vector<std::vector<Node*>> res = rnn->forward(inputs, input_hiddens);
        encoder_states.reserve(layer_num);
        for (uint i = 0; i < layer_num; i++) {
            encoder_states.push_back(res[i].back());
        }
        assert(res.size() == layer_num);
        assert(res[layer_num - 1].size() == num_steps);
        assert(res[layer_num - 1][0]->getShape().colCnt == batch_size);
        return res[layer_num - 1];
    }

    std::vector<Parameters *> Seq2SeqEncoder::get_parameters() {
        std::vector<Parameters *> res;
        std::vector<Parameters *> embedding_params = embedding->get_parameters();
        res.insert(res.end(), embedding_params.begin(), embedding_params.end());
        std::vector<Parameters *> rnn_params = rnn->get_parameters();
        res.insert(res.end(), rnn_params.begin(), rnn_params.end());
        return res;
    }

    Seq2SeqDecoder::Seq2SeqDecoder(
        uint _vocab_size,
        uint _embed_size,
        uint _hidden_num, uint _layer_num,
        DATATYPE sigma, DATATYPE _dropout
    ) : vocab_size(_vocab_size),
        embed_size(_embed_size),
        hidden_num(_hidden_num),
        layer_num(_layer_num),
        dropout(_dropout),
        training(true) {
        assert(layer_num > 0);
        embedding = new Embedding(vocab_size, embed_size);
        rnn = new GRU(embed_size + hidden_num, hidden_num, layer_num, sigma, dropout);
        output_layer = new Liner(hidden_num, vocab_size, sigma);
    }

    Seq2SeqDecoder::~Seq2SeqDecoder() {
        delete output_layer;
        delete rnn;
        delete embedding;
    }

    Node* Seq2SeqDecoder::forward(
        const std::vector<std::vector<uint>> &token_ids,
        Node *ctx,
        const std::vector<Node *> &encoder_states) {
        assert(ctx != nullptr);
        assert(ctx->getShape().rowCnt == hidden_num);
        assert(token_ids.size() > 0);
        assert(ctx->getShape().colCnt == token_ids[0].size());
        assert(encoder_states.size() == layer_num);
        std::vector<Node *> inputs = embedding->forward(token_ids);
        std::vector<Node *> embs_and_context;
        embs_and_context.reserve(inputs.size());
        for (uint i = 0; i < inputs.size(); i++) {
            assert(inputs[i]->getShape().colCnt == ctx->getShape().colCnt);
            embs_and_context.push_back(cat({inputs[i], ctx}, 1));
        }
        std::vector<std::vector<Node*>> res;
        res = rnn->forward(embs_and_context, encoder_states);
        assert(res.size() == layer_num);
        std::vector<Node *> outputs;
        for (auto r : res[layer_num - 1]) {
            outputs.push_back(output_layer->forward(r));
        }
        return cat(outputs);
    }

    std::vector<uint> Seq2SeqDecoder::predict(
                Node *ctx,
                const std::vector<Node *> &encoder_states,
                uint max_len,
                uint bos_id,
                uint eos_id
            ) {
        assert(ctx != nullptr);
        assert(ctx->getShape().rowCnt == hidden_num);
        assert(encoder_states.size() == layer_num);
        std::vector<uint> res;
        std::vector<uint> input = {bos_id};
        std::vector<Node *> input_hiddens = encoder_states;

        for (uint i = 0; i < max_len; i++) {
            std::vector<std::vector<uint>> token_ids = {input};
            std::vector<Node *> inputs = embedding->forward(token_ids);
            std::vector<Node *> embs_and_context;
            assert(inputs.size() == 1);
            assert(inputs[0]->getShape().colCnt == 1);
            embs_and_context.push_back(cat({inputs[0], ctx}, 1));
            
            std::vector<std::vector<Node*>> hiddens;
            hiddens = rnn->forward(embs_and_context, input_hiddens);
            assert(hiddens.size() == layer_num);
            assert(hiddens[layer_num - 1].size() == 1);
            auto output = output_layer->forward(hiddens[layer_num - 1][0]);
            
            assert(output->getShape().colCnt == 1);
            uint token_id = output->get_weight()->argMax()[0];
            res.push_back(token_id);
            if (token_id == eos_id) {
                break;
            }
            input = {token_id};
            input_hiddens.clear();
            for (auto hidden : hiddens) {
                input_hiddens.push_back(hidden[0]);
            }
        }
        return res;
    }

    std::vector<Parameters *> Seq2SeqDecoder::get_parameters() {
        std::vector<Parameters *> res;
        std::vector<Parameters *> embedding_params = embedding->get_parameters();
        res.insert(res.end(), embedding_params.begin(), embedding_params.end());
        std::vector<Parameters *> rnn_params = rnn->get_parameters();
        res.insert(res.end(), rnn_params.begin(), rnn_params.end());
        std::vector<Parameters *> output_params = output_layer->get_parameters();
        res.insert(res.end(), output_params.begin(), output_params.end());
        return res;
    }

    Node* Seq2SeqEncoderDecoder::forward(
        const std::vector<std::vector<uint>> &src_token_ids,
        const std::vector<std::vector<uint>> &tgt_token_ids ) {
        std::vector<Node *> encoder_states;
        auto hiddens = encoder->forward(src_token_ids, encoder_states);
        assert(encoder_states.size() == decoder->get_layer_num());
        assert(encoder_states[0]->getShape().rowCnt == decoder->get_hidden_num());
        auto dec_outputs = decoder->forward(tgt_token_ids, hiddens.back(), encoder_states);
        return dec_outputs;
    }

    std::vector<uint> Seq2SeqEncoderDecoder::predict(
        const std::vector<uint> &src_token_ids,
        uint max_len) {
        std::vector<std::vector<uint>> inputs;
        for (auto token_id : src_token_ids) {
            inputs.push_back({token_id});
        }
        std::vector<Node *> encoder_states;
        auto hiddens = encoder->forward(inputs, encoder_states);
        assert(encoder_states.size() == decoder->get_layer_num());
        assert(encoder_states[0]->getShape().rowCnt == decoder->get_hidden_num());
        assert(encoder_states[0]->getShape().colCnt == 1);
        return decoder->predict(hiddens.back(), encoder_states, max_len, bos_id, eos_id);
    }

    std::vector<Parameters *> Seq2SeqEncoderDecoder::get_parameters() {
        std::vector<Parameters *> res;
        std::vector<Parameters *> encoder_params = encoder->get_parameters();
        res.insert(res.end(), encoder_params.begin(), encoder_params.end());
        std::vector<Parameters *> decoder_params = decoder->get_parameters();
        res.insert(res.end(), decoder_params.begin(), decoder_params.end());
        return res;
    }
} // namespace autograd