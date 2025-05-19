#ifndef MHA_H
#define MHA_H
#include "attention.h"
#include "linear.h"

class MHA {
    public:
        MHA(
            int num_hiddens,
            int _num_heads,
            float dropout = 0.0f,
            bool bias = false
        );
        ~MHA();
    private:
        int num_heads;
        DotProductAttention *attention;
        LazyLinear *w_q;
        LazyLinear *w_k;
        LazyLinear *w_v;
        LazyLinear *w_o;
};

#endif