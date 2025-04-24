#ifndef AUTOGRAD_MLP_H
#define AUTOGRAD_MLP_H

#include "autograd/node.cuh"
#include "autograd/parameters.cuh"

namespace autograd_cuda {

class MLP {
    public:
        MLP(uint _input, const std::vector<uint> &_outputs);
        ~MLP();
        Node *forward(Node *input);
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