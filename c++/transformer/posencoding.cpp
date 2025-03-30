#include "posencoding.h"

#include <cmath>

PosEncoding::PosEncoding(int _max_len, int _num_hidden, DATATYPE _dropout)
: max_len(_max_len), num_hidden(_num_hidden), dropout(_dropout), dropout_layer(nullptr) {
    pos_encoding.reserve(max_len);
    for (int i = 0; i < max_len; i++) {
        Matrix *pe = new Matrix(Shape(num_hidden, 1));
        for (int j = 0; j < num_hidden; j++) {
            if (j % 2 == 0) {
                (*pe)[j][0] = sin(i / pow(10000, j / num_hidden));
            } else {
                (*pe)[j][0] = cos(i / pow(10000, j / num_hidden));
            }
        }
        pos_encoding_matrics.push_back(pe);
        pos_encoding.push_back(new autograd::Node(pe, true));
    }
    for (int i = 0; i < max_len; i++) {
        std::cout << *pos_encoding[i]->get_weight() << std::endl;
    }
    if (dropout > 0) {
        dropout_layer = new autograd::Dropout(dropout);
    }
}

PosEncoding::~PosEncoding() {
    if (dropout > 0 && dropout_layer != nullptr) {
        delete dropout_layer;
    }
    for (auto pe : pos_encoding) {
        delete pe;
    }
    for (auto pe : pos_encoding_matrics) {
        delete pe;
    }
}

std::vector<autograd::Node *> PosEncoding::forward(const std::vector<autograd::Node *> &x) {
    std::vector<autograd::Node *> res;
    for (uint i = 0; i < x.size(); i++) {
        res.push_back(*(x[i]) + pos_encoding[i]);
    }
    if (dropout > 0) {
        res = dropout_layer->forward(res);
    }
    return res;
}