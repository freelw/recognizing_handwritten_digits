#ifndef AUTOGRAD_MLP_H
#define AUTOGRAD_MLP_H

#include "autograd/node.h"
#include "autograd/parameters.h"

namespace autograd {

class MLP {
    public:
        MLP(uint _input, const std::vector<uint> &_outputs);
        ~MLP();
        Matrix *forward(Matrix *input);
        DATATYPE backward(Matrix *input, const std::vector<uint> &labels);
        std::vector<Parameters*> get_parameters();
    private:
        Matrix *mW1;
        Matrix *mb1;
        Matrix *mW2;
        Matrix *mb2;
        Node *W1;
        Node *b1;
        Node *W2;
        Node *b2;
        Parameters *PW1;
        Parameters *Pb1;
        Parameters *PW2;
        Parameters *Pb2;
};
    
} // namespace autograd


#endif