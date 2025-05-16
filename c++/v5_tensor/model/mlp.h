#ifndef V5_MLP_H
#define V5_MLP_H

#include "module/dropout.h"
#include "optimizers/parameter.h"

class MLP {
    public:
        MLP(
            int32_t _input,
            const std::vector<int32_t> &_outputs,
            float dropout_p,
            bool const_weight = false
        );
        ~MLP();
        std::vector<Parameter*> get_parameters();
        graph::Node *forward(graph::Node *input);
    private:
        Tensor *w1;
        Tensor *w2;
        Tensor *bias1;
        Tensor *bias2;

        graph::Node *nw1;
        graph::Node *nw2;
        graph::Node *nb1;
        graph::Node *nb2;

        Parameter *pw1;
        Parameter *pw2;
        Parameter *pb1;
        Parameter *pb2;
        Dropout *dropout;
};

#endif
