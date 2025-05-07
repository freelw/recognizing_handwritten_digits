#ifndef V5_MLP_H
#define V5_MLP_H


#include "optimizers/parameter.h"

class MLP {
    public:
        MLP(int32_t _input, const std::vector<int32_t> &_outputs);
        ~MLP() = default;
};

#endif
