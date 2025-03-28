#include "attention.h"

DotProductAttetion::DotProductAttetion(DATATYPE _dropout)
    : dropout(_dropout), dropout_layer(nullptr), is_training(true) {
    assert (dropout >= 0 && dropout < 1);
    if (dropout > 0) {
        dropout_layer = new autograd::Dropout(dropout);
    }
}

DotProductAttetion::~DotProductAttetion() {
    if (dropout > 0 && dropout_layer != nullptr) {
        delete dropout_layer;
    }
}

void mask(autograd::Node *node, uint valid_len) {
    for (uint i = valid_len; i < node->get_weight()->getShape().rowCnt; i++) {
        for (uint j = 0; j < node->get_weight()->getShape().colCnt; j++) {
            (*node->get_weight())[i][j] = -1e6;
        }
    }
}

std::vector<autograd::Node *> DotProductAttetion::forward(
    const std::vector<autograd::Node *> &Q,
    const std::vector<autograd::Node *> &K,
    const std::vector<autograd::Node *> &V,
    const std::vector<uint> &valid_lens
) {
    assert (K.size() == V.size());
    std::vector<autograd::Node *> res;
    std::vector<autograd::Node *> scores;
    for (size_t i = 0; i < Q.size(); i++) {
        autograd::Node *q = Q[i];
        autograd::Node *k = K[i];
        autograd::Node *score = q->Transpose()->at(k)->Transpose();
        score = score->Div(sqrt(k->getShape().rowCnt));
        mask(score, valid_lens[i]);
        score = score->Softmax();
        scores.push_back(score);
    }

    if (dropout > 0 && training()) {
        scores = dropout_layer->forward(scores);
    }

    for (size_t i = 0; i < V.size(); i++) {
        autograd::Node *att = V[i]->at(scores[i]);
        res.push_back(att);
    }
    return res;
}

MultiHeadAttention::MultiHeadAttention(
    uint _num_heads,
    uint _num_hidden,
    DATATYPE dropout) 
    : num_heads(_num_heads), num_hidden(_num_hidden), attention(nullptr), is_training(true) {
    assert(_num_hidden % num_heads == 0);
    attention = new DotProductAttetion(dropout);
    Wq = new autograd::LazyLiner(num_hidden, false);
    Wk = new autograd::LazyLiner(num_hidden, false);
    Wv = new autograd::LazyLiner(num_hidden, false);
    Wo = new autograd::LazyLiner(num_hidden, false);
}

MultiHeadAttention::~MultiHeadAttention() {
    delete Wo;
    delete Wv;
    delete Wk;
    delete Wq;
    delete attention;
}

std::vector<autograd::Node *> MultiHeadAttention::forward(
    const std::vector<autograd::Node *> &queries,
    const std::vector<autograd::Node *> &keys,
    const std::vector<autograd::Node *> &values,
    const std::vector<uint> &valid_lens
) {
    assert(num_hidden % num_heads == 0);
    assert(queries.size() == keys.size());
    assert(queries.size() == values.size());
    uint step = num_hidden / num_heads;

    std::vector<autograd::Node *> split_queries;
    std::vector<autograd::Node *> split_keys;
    std::vector<autograd::Node *> split_values;
    std::vector<uint> split_valid_lens;

    split_queries.reserve(queries.size() * num_heads);
    split_keys.reserve(keys.size() * num_heads);
    split_values.reserve(values.size() * num_heads);
    split_valid_lens.reserve(valid_lens.size() * num_heads);

    for (auto & q : queries) {
        std::vector<autograd::Node *> tmp = Wq->forward(q)->split(1, step);
        split_queries.insert(split_queries.end(), tmp.begin(), tmp.end());
    }
    for (auto & k : keys) {
        std::vector<autograd::Node *> tmp = Wk->forward(k)->split(1, step);
        split_keys.insert(split_keys.end(), tmp.begin(), tmp.end());   
    }
    for (auto & v : values) {
        std::vector<autograd::Node *> tmp = Wv->forward(v)->split(1, step);
        split_values.insert(split_values.end(), tmp.begin(), tmp.end());
    }
    for (auto & len : valid_lens) {
        for (uint i = 0; i < num_heads; i++) {
            split_valid_lens.push_back(len);
        }
    }

    std::vector<autograd::Node *> atts = attention->forward(split_queries, split_keys, split_values, split_valid_lens);
    assert(atts.size() == split_queries.size());
    std::vector<autograd::Node *> res;
    res.reserve(queries.size());
    for (uint i = 0; i < atts.size(); i += num_heads) {
        std::vector<autograd::Node *> tmp;
        for (uint j = 0; j < num_heads; j++) {
            tmp.push_back(atts[i + j]);
        }
        autograd::Node *att = autograd::cat(tmp, 1);
        res.push_back(Wo->forward(att));
    }
    return res;
}

std::vector<autograd::Parameters *> MultiHeadAttention::get_parameters() {
    std::vector<autograd::Parameters *> res;
    auto q_params = Wq->get_parameters();
    auto k_params = Wk->get_parameters();
    auto v_params = Wv->get_parameters();
    auto o_params = Wo->get_parameters();
    res.insert(res.end(), q_params.begin(), q_params.end());
    res.insert(res.end(), k_params.begin(), k_params.end());
    res.insert(res.end(), v_params.begin(), v_params.end());
    res.insert(res.end(), o_params.begin(), o_params.end());
    return res;
}

