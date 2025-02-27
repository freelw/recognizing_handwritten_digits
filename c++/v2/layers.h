#ifndef LAYERS_H
#define LAYERS_H

#include <vector>
#include "matrix/matrix.h"

class Context {

};

class Layer {
    public:
        Layer() {}
        virtual ~Layer() {}
        virtual Matrix *forward(Context *, Matrix *input) = 0;
        virtual Matrix *backward(Context *, Matrix *grad) = 0;
        virtual Context *init() = 0;
        virtual void release(Context *) = 0;
};

class ReluContext: public Context {
    public:
        Matrix *input;
};

class Relu: public Layer {
    public:
        Relu() {}
        virtual ~Relu() {}
        virtual Matrix *forward(Context *, Matrix *input) = 0;
        virtual Matrix *backward(Context *, Matrix *grad) = 0;
        virtual Context *init() = 0;
        virtual void release(Context *) = 0;
};

struct CrosEntropyInfo {
    DATATYPE ez, sum, max;
};
class CrossEntropyLossContext: public Context {
    public:
        std::vector<CrosEntropyInfo> info;
        Matrix *input;
};

class CrossEntropyLoss: public Layer {
    public:
        CrossEntropyLoss(const std::vector<uint> & _lalels): labels(_lalels) {}
        ~CrossEntropyLoss() {}
        virtual Matrix *forward(Context *, Matrix *input);
        virtual Matrix *backward(Context *, Matrix *grad);
        virtual Context *init();
        virtual void release(Context *);
    private:
        std::vector<uint> labels;
};

#endif