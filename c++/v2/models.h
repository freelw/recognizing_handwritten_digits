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
    private:
        std::vector<Layer*> layers;
        uint input;
        std::vector<uint> outputs;
        std::vector<Context*> ctxs;
        Matrix *forward(Matrix *input);
        void backword(Matrix *input, const std::vector<uint> &labels);
};

#endif