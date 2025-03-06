#ifndef LAYERS_H
#define LAYERS_H

#include <assert.h>
#include <vector>
#include "matrix/matrix.h"

class Context {

};

class Parameters {
    public:
        Parameters(Shape shape) : grad(nullptr), t(0) {
            w = new Matrix(shape);
            m = new Matrix(shape);
            v = new Matrix(shape);
        }
        ~Parameters() {
            assert(w != nullptr);
            delete w;
            delete m;
            delete v;
        }
        void set_grad(Matrix * _grad) {
            assert(grad == nullptr);
            grad = _grad;
        }
        void inc_grad(Matrix * _grad) {
            if (grad == nullptr) {
                grad = _grad;
            } else {
                grad->checkShape(*_grad);
                *grad += *_grad;
            }
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
        Matrix *get_m() {
            return m;
        }
        Matrix *get_v() {
            return v;
        }
        int get_t() {
            return t;
        }
        void inc_t() {
            t++;
        }
        friend std::ostream & operator<<(std::ostream &output, const Parameters &p) {
            output << std::endl << "weight : " << endl << "\t";
            Shape shape = p.w->getShape();
            for (uint i = 0; i < shape.rowCnt; ++ i) {
                for (uint j = 0; j < shape.colCnt; ++ j) {
                    output << (*p.w)[i][j] << " ";
                }
                output << endl << "\t";
            }
            output << std::endl << "grad : " << endl << "\t";
            for (uint i = 0; i < shape.rowCnt; ++ i) {
                for (uint j = 0; j < shape.colCnt; ++ j) {
                    output << (*p.grad)[i][j] << " ";
                }
                output << endl << "\t";
            }
            return output;
        }
    private:
        Parameters(const Parameters&);    
        Parameters& operator=(const Parameters&);
    private:
        Matrix *w;
        Matrix *grad;
        // m v for adam opt
        Matrix *m;
        Matrix *v;
        int t;
};

class Layer {
    public:
        Layer() {}
        virtual ~Layer() {}
        virtual Matrix *forward(Context *, Matrix *input) = 0;
        virtual Matrix *backward(Context *, Matrix *grad) = 0;
        virtual Context *init() = 0;
        virtual void release(Context *) = 0;
        virtual std::vector<Parameters*> get_parameters() {
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
        Liner(uint i, uint o, DATATYPE sigma, bool rand);
        virtual ~Liner();
        virtual Matrix *forward(Context *, Matrix *input);
        virtual Matrix *backward(Context *, Matrix *grad);
        virtual Context *init();
        virtual void release(Context *);
        virtual std::vector<Parameters*> get_parameters();
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
    DATATYPE sum, max, zt;
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

class RnnContext {
    public:
        std::vector<Matrix*> inputs;
        std::vector<Matrix*> hiddens;
        std::vector<Matrix*> states;
        void clear() {
            inputs.clear();
            hiddens.clear();
            states.clear();
        }
};

struct RnnRes {
    std::vector<Matrix *> states;
};

class Rnn {
    public:
        Rnn(uint i, uint h, DATATYPE _sigma, bool _rand);
        virtual ~Rnn();
        virtual RnnRes forward(RnnContext *, const std::vector<Matrix*> &inputs, Matrix *hidden);
        virtual Matrix *backward(RnnContext *, const std::vector<Matrix *> &grad_hiddens_vec);
        virtual RnnContext *init();
        virtual void release(RnnContext *);
        virtual std::vector<Parameters*> get_parameters();
        virtual void zero_grad();
        DATATYPE get_sigma() {
            return sigma;
        }
        uint get_hidden_num() {
            return hidden_num;
        }
    private:
        uint input_num;
        uint hidden_num;
        DATATYPE sigma;
        Parameters *wxh;
        Parameters *whh;
        Parameters *bh;
        bool rand;
};

#endif