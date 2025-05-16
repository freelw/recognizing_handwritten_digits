#include "dropout.h"
extern bool g_training;

graph::Node *Dropout::forward(graph::Node *x) {
    if (g_training) {
        // fix me
        return x;
    } else {
        return x;
    }
}