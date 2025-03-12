#ifndef AUTOGRAD_RNNLM_H
#define AUTOGRAD_RNNLM_H

#include "autograd/node.h"
#include "autograd/parameters.h"

namespace autograd {

class Rnn {
    public:
        Rnn(uint input_num, uint _hidden_num, DATATYPE sigma);
        ~Rnn();
        std::vector<Node *> forward(const std::vector<Node *> &inputs, Node *prev_hidden);
        std::vector<Parameters *> get_parameters();
        uint get_hidden_num() { return hidden_num; }
    private:

        uint hidden_num;
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
        RnnLM(Rnn *_rnn, uint vocab_size);
        ~RnnLM();
        Node *forward(std::vector<Node *> inputs);
        Node *output_layer(Node *hidden);
        std::string predict(const std::string &prefix, uint num_preds);
        std::vector<Parameters *> get_parameters();
    private:
        Rnn *rnn;
        uint vocab_size;
        
        Matrix *mW;
        Matrix *mb;
        Node *W;
        Node *b;
        Parameters *PW;
        Parameters *Pb;
};
} // namespace autograd
#endif