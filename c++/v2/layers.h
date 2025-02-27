#ifndef LAYERS_H
#define LAYERS_H

#include <assert.h>
#include <vector>
#include "matrix/matrix.h"


class Context {

};

class Parameters {
    public:
        Parameters() : w(nullptr), grad(nullptr), m(nullptr), v(nullptr) {}
        Parameters(const Parameters & o) : w(o.w), grad(o.grad), m(o.m), v(o.v) {}
        ~Parameters() {
            delete m;
            delete v;
        }
        void set_weight(Matrix * _w) {
            assert(w == nullptr);
            w = _w;
            m = new Matrix(w->getShape());
            v = new Matrix(w->getShape());
        }
        void set_grad(Matrix * _grad) {
            assert(grad == nullptr);
            grad = _grad;
        }
        void zero_grad() {
            grad = nullptr;
        }
        Matrix *get_weight() {
            return w;
        }
        Matrix *get_grad() {
            return grad;
        }
    private:
        Matrix *w;
        Matrix *grad;
        // m v for adam opt
        Matrix *m;
        Matrix *v;
};

class Layer {
    public:
        Layer() {}
        virtual ~Layer() {}
        virtual Matrix *forward(Context *, Matrix *input) = 0;
        virtual Matrix *backward(Context *, Matrix *grad) = 0;
        virtual Context *init() = 0;
        virtual void release(Context *) = 0;
        virtual std::vector<Parameters> get_parameters() {
            return {};
        }
        virtual void zero_grad() {}
};

class LinerContext: public Context {
    public:
        Matrix *input;
};

class Liner: public Layer {
    public:
        Liner(uint i, uint o);
        virtual ~Liner() {}
        virtual Matrix *forward(Context *, Matrix *input);
        virtual Matrix *backward(Context *, Matrix *grad);
        virtual Context *init();
        virtual void release(Context *);
        virtual std::vector<Parameters> get_parameters();
        virtual void zero_grad();
    private:
        uint input_num;
        uint output_num;
        Parameters *weigt;
        Parameters *bias;
};

class ReluContext: public Context {
    public:
        Matrix *input;
};

class Relu: public Layer {
    public:
        Relu() {}
        virtual ~Relu() {}
        virtual Matrix *forward(Context *, Matrix *input);
        virtual Matrix *backward(Context *, Matrix *grad);
        virtual Context *init();
        virtual void release(Context *);
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