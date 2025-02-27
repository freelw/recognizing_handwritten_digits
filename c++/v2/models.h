#ifndef MODELS_H
#define MODELS_H

#include "layers.h"

class MLP {
    public:
        MLP(uint _input, const std::vector<uint> &_outputs);
        ~MLP() {
            destroy();
        }
        void init();
        void destroy();
        Matrix *forward(Matrix *input);
        DATATYPE backward(Matrix *input, const std::vector<uint> &labels);
        std::vector<Parameters> get_parameters();
        void zero_grad();
    private:
        std::vector<Layer*> layers;
        uint input;
        std::vector<uint> outputs;
        std::vector<Context*> ctxs;
};

#endif