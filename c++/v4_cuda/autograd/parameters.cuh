#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "autograd/node.cuh"

namespace autograd_cuda {
    class Parameters {
        public:
            Parameters(Node *node);
            ~Parameters();
            void zero_grad();
            Matrix *get_weight();
            Matrix *get_grad();
            Matrix *get_m();
            Matrix *get_v();
            int get_t();
            void inc_t();
            std::string serialize();
            void deserialize(char *buffer);
            bool require_grad();
        private:
            void sync();
        private:
            Node *w;
            Matrix *m;
            Matrix *v;
            int t;
    };
};
#endif