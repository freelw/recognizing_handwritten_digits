#ifndef LAYERS_H
#define LAYERS_H


#include "matrix/matrix.h"

class Context {

};

class Layer {
    public:
        Layer() {}
        virtual ~Layer() {}
        virtual Matrix *forward(Context *, Matrix *input) = 0;
        virtual Matrix *backward(Context *, Matrix *grad) = 0;
};

class CrossEntropyLoss: public Layer {
    public:
        CrossEntropyLoss() {}
        ~CrossEntropyLoss() {}
        virtual Matrix *forward(Context *, Matrix *input);
        virtual Matrix *backward(Context *, Matrix *grad);
};

#endif