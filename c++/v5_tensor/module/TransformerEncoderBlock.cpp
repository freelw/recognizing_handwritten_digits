#include "TransformerEncoderBlock.h"

TransformerEncoderBlock::TransformerEncoderBlock(
    int num_hiddens, int ffn_num_hiddens, int num_heads,
    float dropout, bool bias) {

    attention = new MHA(num_hiddens, num_heads, dropout, bias);
    addnorm1 = new AddNorm(num_hiddens, dropout);
    ffn = new PositionWiseFFN(ffn_num_hiddens, num_hiddens);
    addnorm2 = new AddNorm(num_hiddens, dropout);
}

TransformerEncoderBlock::~TransformerEncoderBlock() {
    delete attention;
    delete addnorm1;
    delete ffn;
    delete addnorm2;
}

graph::Node *TransformerEncoderBlock::forward(
    graph::Node *x, Tensor *valid_lens) {
    auto y = attention->forward(x, x, x, valid_lens);
    // gCreateAction(new DbgPrintAction(
    //     y->get_tensor(),
    //     "TransformerEncoderBlock attention res y : "
    // ));
    y = addnorm1->forward(x, y);
    // gCreateAction(new DbgPrintAction(
    //     y->get_tensor(),
    //     "TransformerEncoderBlock addnorm1 res y : "
    // ));
    auto z = ffn->forward(y);
    // gCreateAction(new DbgPrintAction(
    //     z->get_tensor(),
    //     "TransformerEncoderBlock ffn res z : "
    // ));
    auto res = addnorm2->forward(y, z);
    // gCreateAction(new DbgPrintAction(
    //     res->get_tensor(),
    //     "TransformerEncoderBlock addnorm2 res : "
    // ));
    return res;
}

std::vector<Parameter *> TransformerEncoderBlock::get_parameters() {
    std::vector<Parameter *> params;
    auto attention_params = attention->get_parameters();
    auto addnorm1_params = addnorm1->get_parameters();
    auto ffn_params = ffn->get_parameters();
    auto addnorm2_params = addnorm2->get_parameters();

    params.insert(params.end(), attention_params.begin(), attention_params.end());
    params.insert(params.end(), addnorm1_params.begin(), addnorm1_params.end());
    params.insert(params.end(), ffn_params.begin(), ffn_params.end());
    params.insert(params.end(), addnorm2_params.begin(), addnorm2_params.end());

    return params;
}