#ifndef AUTOGRAD_RNNLM_H
#define AUTOGRAD_RNNLM_H

#include "autograd/node.h"
#include "autograd/parameters.h"

namespace autograd {

    class LSTM {
        public:
            LSTM(uint input_num, uint _hidden_num, DATATYPE sigma);
            ~LSTM();
            std::vector<std::pair<Node *, Node*>> forward(const std::vector<Node *> &inputs, Node *prev_hidden, Node *cell);
            std::vector<Parameters *> get_parameters();
            uint get_hidden_num() { return hidden_num; }
        private:
            uint hidden_num;
            Matrix *mWxi;
            Matrix *mWhi;
            Matrix *mBi;
            Matrix *mWxf;
            Matrix *mWhf;
            Matrix *mBf;
            Matrix *mWxo;
            Matrix *mWho;
            Matrix *mBo;
            Matrix *mWxc;
            Matrix *mWhc;
            Matrix *mBc;

            Node *Wxi;
            Node *Whi;
            Node *Bi;
            Node *Wxf;
            Node *Whf;
            Node *Bf;
            Node *Wxo;
            Node *Who;
            Node *Bo;
            Node *Wxc;
            Node *Whc;
            Node *Bc;

            Parameters *PWxi;
            Parameters *PWhi;
            Parameters *PBi;
            Parameters *PWxf;
            Parameters *PWhf;
            Parameters *PBf;
            Parameters *PWxo;
            Parameters *PWho;
            Parameters *PBo;
            Parameters *PWxc;
            Parameters *PWhc;
            Parameters *PBc;
    };

    class RnnLM {
        public:
            RnnLM(LSTM *_rnn, uint vocab_size);
            ~RnnLM();
            Node *forward(std::vector<Node *> inputs);
            Node *output_layer(Node *hidden);
            std::string predict(const std::string &prefix, uint num_preds);
            std::vector<Parameters *> get_parameters();
        private:
            LSTM *rnn;
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