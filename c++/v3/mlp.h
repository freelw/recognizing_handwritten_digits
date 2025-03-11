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
        Parameters *W1;
        Parameters *b1;
        Parameters *W2;
        Parameters *b2;
};
    
} // namespace autograd


#endif