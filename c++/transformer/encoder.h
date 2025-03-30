#ifndef ENCODER_H
#define ENCODER_H

#include "autograd/node.h"
#include "dropout.h"

class EncoderBlock {

    public:
        EncoderBlock(int _num_hidden, int _num_heads, DATATYPE _dropout);
        ~EncoderBlock();
        std::vector<autograd::Node *> forward(const std::vector<autograd::Node *> &x);
        void train(bool _training);
        bool is_training();
    private:
        int num_hidden;
        int num_heads;
        bool training;
        DATATYPE dropout;
        
        autograd::Dropout *dropout_layer;
};

#endif