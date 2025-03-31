#include "encoder.h"

EncoderBlock::EncoderBlock(uint _num_hidden, uint _ffn_num_hiddens, uint _num_heads, DATATYPE dropout, bool _bias)
    : num_hidden(_num_hidden),
    ffn_num_hiddens(_ffn_num_hiddens),
    num_heads(_num_heads),
    training(true),
    bias(_bias)
{
    attention = new MultiHeadAttention(num_heads, num_hidden, dropout, bias);
    ffn = new PositionwiseFFN(_ffn_num_hiddens, num_hidden);
    addnorm1 = new AddNorm(num_hidden, dropout);
    addnorm2 = new AddNorm(num_hidden, dropout);
}

EncoderBlock::~EncoderBlock() {
    delete attention;
    delete ffn;
    delete addnorm1;
    delete addnorm2;
}

std::vector<autograd::Node *> EncoderBlock::forward(
    const std::vector<autograd::Node *> &x,
    const std::vector<uint> &valid_lens) {
    std::vector<autograd::Node *> y = addnorm1->forward(x, attention->forward(x, x, x, valid_lens));
    return addnorm2->forward(y, ffn->forward(y));
}

std::vector<autograd::Parameters *> EncoderBlock::get_parameters() {
    std::vector<autograd::Parameters *> res;
    auto p1 = attention->get_parameters();
    auto p2 = ffn->get_parameters();
    auto p3 = addnorm1->get_parameters();
    auto p4 = addnorm2->get_parameters();
    res.insert(res.end(), p1.begin(), p1.end());
    res.insert(res.end(), p2.begin(), p2.end());
    res.insert(res.end(), p3.begin(), p3.end());
    res.insert(res.end(), p4.begin(), p4.end());
    return res;
}

void EncoderBlock::train(bool _training) {
    training = _training;
    attention->train(_training);
    addnorm1->train(_training);
    addnorm2->train(_training);
}
bool EncoderBlock::is_training() {
    return training;
}