#include "posencoding.h"
#include "graph/actions.h"

PosEncoding::PosEncoding(int _max_len, int _num_hidden, float p)
    :max_len(_max_len), num_hidden(_num_hidden) {
    pos_enc = allocTensor({max_len, num_hidden}, "pos_enc");
    gCreateAction(
        new PosEncodingAction(
            pos_enc
        )
    );
    dropout = new Dropout(p);
}

PosEncoding::~PosEncoding() {
    assert(dropout != nullptr);
    delete dropout;
}

graph::Node *PosEncoding::forward(graph::Node *input) {
    assert(input->get_tensor()->get_dim() == 2);
    auto shape = input->get_tensor()->get_shape();
    assert(shape.size() == 2);
    assert(shape[1] == num_hidden);
    Tensor *pe = allocTensorView(
        pos_enc,
        {shape[0], num_hidden},
        {num_hidden, 1},
        "pos_enc_view",
        0
    );
    auto npe = graph::allocNode(pe);
    return dropout->forward(input->add(npe));
}
