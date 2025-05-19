#include "mha.h"

MHA::MHA(
    int num_hiddens,
    int _num_heads,
    float dropout,
    bool bias
) : num_heads(_num_heads) {
    attention = new DotProductAttention(dropout);

}