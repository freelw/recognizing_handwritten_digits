#ifndef LINER_H
#define LINER_H

#include "autograd/node.h"
#include "autograd/parameters.h"

namespace autograd {
    class Linear {
        public:
            Linear(uint input_num, uint output_num, bool _bias = true);
            ~Linear();
            Node *forward(Node *input);
            std::vector<Node *> forward(const std::vector<Node *> &input);
            std::vector<Parameters *> get_parameters();
        private:
            bool bias;
            Matrix *mW;
            Matrix *mb;
            Node *W;
            Node *b;
            Parameters *PW;
            Parameters *Pb;
    };

    class LazyLinear {
        public:
            LazyLinear(uint _output_num, bool _bias = true);
            ~LazyLinear();
            Node *forward(Node *input);
            std::vector<Node *> forward(const std::vector<Node *> &input);
            std::vector<Parameters *> get_parameters();
        private:
            uint output_num;
            bool bias;
            Linear *linear;
    };
} // namespace autograd
#endif