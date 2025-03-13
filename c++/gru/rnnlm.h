#ifndef AUTOGRAD_RNNLM_H
#define AUTOGRAD_RNNLM_H

#include "autograd/node.h"
#include "autograd/parameters.h"

namespace autograd {

    class GRU {
        public:
            GRU(uint input_num, uint _hidden_num, DATATYPE sigma);
            ~GRU();
            std::vector<Node*> forward(const std::vector<Node *> &inputs, Node *prev_hidden);
            std::vector<Parameters *> get_parameters();
            uint get_hidden_num() { return hidden_num; }
        private:
            uint hidden_num;
            Matrix *mWxr;
            Matrix *mWhr;
            Matrix *mBr;
            Matrix *mWxz;
            Matrix *mWhz;
            Matrix *mBz;
            Matrix *mWxh;
            Matrix *mWhh;
            Matrix *mBh;

            Node *Wxr;
            Node *Whr;
            Node *Br;
            Node *Wxz;
            Node *Whz;
            Node *Bz;
            Node *Wxh;
            Node *Whh;
            Node *Bh;

            Parameters *PWxr;
            Parameters *PWhr;
            Parameters *PBr;
            Parameters *PWxz;
            Parameters *PWhz;
            Parameters *PBz;
            Parameters *PWxh;
            Parameters *PWhh;
            Parameters *PBh;
    };

    class RnnLM {
        public:
            RnnLM(GRU *_rnn, uint vocab_size);
            ~RnnLM();
            Node *forward(std::vector<Node *> inputs);
            Node *output_layer(Node *hidden);
            std::string predict(const std::string &prefix, uint num_preds);
            std::vector<Parameters *> get_parameters();
        private:
            GRU *rnn;
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