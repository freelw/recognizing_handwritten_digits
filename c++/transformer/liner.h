#ifndef LINER_H
#define LINER_H

#include "autograd/node.h"
#include "autograd/parameters.h"

namespace autograd {
    class Liner {
        public:
            Liner(uint input_num, uint output_num, bool _bias = true);
            ~Liner();
            Node *forward(Node *input);
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
} // namespace autograd
#endif