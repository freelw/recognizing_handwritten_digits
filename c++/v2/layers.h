#ifndef LAYERS_H
#define LAYERS_H

#include <assert.h>
#include <vector>
#include "matrix/matrix.h"

class Context {

};

class Parameters {
    public:
        Parameters(Shape shape) : grad(nullptr) {
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
        // void set_weight(Matrix * _w) {
        //     assert(w == nullptr);
        //     w = _w;
        //     m = new Matrix(w->getShape());
        //     v = new Matrix(w->getShape());
        // }
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
        Liner(uint i, uint o, bool);
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