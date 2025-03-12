#ifndef AUTOGRAD_RNNLM_H
#define AUTOGRAD_RNNLM_H

#include "autograd/node.h"
#include "autograd/parameters.h"

namespace autograd {

class Rnn {
    public:
        Rnn(uint input_num, uint hidden_num);
        ~Rnn();
        Node *forward(const std::vector<Node *> &input, Node *prev_hidden);
        std::vector<Parameters *> get_parameters();
    private:
        Matrix *mWxh;
        Matrix *mWhh;
        Matrix *mbh;
        
        Node *Wxh;
        Node *Whh;
        Node *bh;

        Parameters *PWxh;
        Parameters *PWhh;
        Parameters *Pbh;
};

class RnnLM {
    public:
        RnnLM(Rnn *rnn, uint vocab_size);
        ~RnnLM();
        Node *forward(Node *ctx, std::vector<Node *> inputs);
        Node *predict(std::string prefix, uint max_len);
        std::vector<Parameters *> get_parameters();
};
} // namespace autograd
#endif