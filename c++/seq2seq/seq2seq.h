#ifndef AUTOGRAD_SEQ2SEQ_H
#define AUTOGRAD_SEQ2SEQ_H

#include "autograd/node.h"
#include "autograd/parameters.h"
#include <random>
#include <chrono>

namespace autograd {

    class Dropout {
        public:
            Dropout(DATATYPE _dropout);
            ~Dropout() {}
            std::vector<Node *> forward(const std::vector<Node *> &inputs);
        private:
            DATATYPE dropout;
            std::mt19937 gen;
            std::uniform_real_distribution<> dis;
    };
    class Embedding {
        public:
            Embedding(uint vocab_size, uint hidden_num);
            ~Embedding();
            std::vector<Node *> forward(const std::vector<std::vector<uint>> &inputs);
            std::vector<Parameters *> get_parameters();
        private:
            uint vocab_size;
            uint hidden_num;
            std::vector<Matrix *> mW;
            std::vector<Node *> W;
            std::vector<Parameters *> PW;
    };

    class GRULayer {
        public:
            GRULayer(uint input_num, uint _hidden_num, DATATYPE sigma);
            ~GRULayer();
            std::vector<Node *> forward(const std::vector<Node *> &inputs, Node *hidden);
            std::vector<Parameters *> get_parameters();
            uint get_hidden_num() { return hidden_num; }
        private:
            uint hidden_num;
            DATATYPE dropout;
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

    class GRU {
        public:
            GRU(
                uint input_num,
                uint _hidden_num,
                uint _layer_num,
                DATATYPE sigma,
                DATATYPE _dropout
            );
            ~GRU();
            std::vector<std::vector<Node*>> forward(
                const std::vector<Node *> &inputs,
                const std::vector<Node *> &prev_hidden
            );
            std::vector<Parameters *> get_parameters();
            uint get_hidden_num() { return hidden_num; }
            uint get_layer_num() { return layer_num; }
            void train(bool _training) { training = _training; }
            bool is_training() { return training; }
        private:
            uint hidden_num;
            uint layer_num;
            DATATYPE dropout;
            std::vector<GRULayer *> layers;
            bool training;
    };

    class Seq2SeqEncoder {
        public:
            Seq2SeqEncoder(
                uint _vocab_size,
                uint _embed_size,
                uint _hidden_num,
                uint _layer_num,
                DATATYPE sigma,
                DATATYPE _dropout
            );
            ~Seq2SeqEncoder();
            std::vector<std::vector<Node*>> forward(
                const std::vector<std::vector<uint>> &token_ids
            );
            std::vector<Parameters *> get_parameters();
            uint get_hidden_num() { return hidden_num; }
            uint get_layer_num() { return layer_num; }
            void train(bool _training) { training = _training; }
            bool is_training() { return training; }
        private:
            uint vocab_size;
            uint embed_size;
            uint hidden_num;
            uint layer_num;
            DATATYPE dropout;
            std::vector<GRULayer *> layers;
            bool training;
            Embedding *embedding;
    };


    class Seq2SeqDecoder {};


    class Seq2SeqEncoderDecoder {
        public:
            Seq2SeqEncoderDecoder() {}
            ~Seq2SeqEncoderDecoder() {}
            std::vector<std::vector<Node*>> forward(
                const std::vector<Node *> &inputs
            );
            std::vector<Parameters *> get_parameters();
            
        private:
            
    };

    
} // namespace autograd
#endif