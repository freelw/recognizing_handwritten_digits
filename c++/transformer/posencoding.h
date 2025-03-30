#ifndef POS_ENCODING_H
#define POS_ENCODING_H

#include "autograd/node.h"
#include "dropout.h"

class PosEncoding {
    public:
        PosEncoding(int _max_len, int _num_hidden, DATATYPE _dropout);
        ~PosEncoding();
        std::vector<autograd::Node *> forward(const std::vector<autograd::Node *> &x);
    private:
        int max_len;
        int num_hidden;
        DATATYPE dropout;
        std::vector<Matrix *> pos_encoding_matrics;
        std::vector<autograd::Node *> pos_encoding;
        autograd::Dropout *dropout_layer;
        
};

#endif