#ifndef LAYERS_H
#define LAYERS_H

#include <assert.h>
#include <vector>
#include "matrix/matrix.h"
#include "parameters/parameters.h"

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
    DATATYPE sum, max;
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

class RnnBase {
    public:
        RnnBase() {};
        virtual ~RnnBase() {};
        virtual RnnRes forward(RnnContext *, const std::vector<Matrix*> &inputs, Matrix *hidden, Matrix *cell) = 0;
        virtual Matrix *backward(
            RnnContext *,
            const std::vector<Matrix *> &grad_hiddens_vec,
            const std::vector<Matrix *> &grad_cells_vec) = 0;
        virtual RnnContext *init() = 0;
        virtual void release(RnnContext *) = 0;
        virtual std::vector<Parameters*> get_parameters() = 0;
        virtual void zero_grad() = 0;
        virtual uint get_hidden_num() = 0;
};

class Rnn: public RnnBase {
    public:
        Rnn(uint i, uint h, DATATYPE _sigma, bool _rand);
        virtual ~Rnn();
        virtual RnnRes forward(RnnContext *, const std::vector<Matrix*> &inputs, Matrix *hidden, Matrix *cell);
        virtual Matrix *backward(
            RnnContext *, 
            const std::vector<Matrix *> &grad_hiddens_vec,
            const std::vector<Matrix *> &grad_cells_vec);
        virtual RnnContext *init();
        virtual void release(RnnContext *);
        virtual std::vector<Parameters*> get_parameters();
        virtual void zero_grad();
        DATATYPE get_sigma() {
            return sigma;
        }
        virtual uint get_hidden_num() {
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

class LSTM: public RnnBase {
    public:
        LSTM(uint i, uint h, DATATYPE _sigma, bool _rand);
        virtual ~LSTM();
        virtual RnnRes forward(
            RnnContext *, const std::vector<Matrix*> &inputs,
            Matrix *hidden, Matrix *cell);
        virtual Matrix *backward(
            RnnContext *,
            const std::vector<Matrix *> &grad_hiddens_vec,
            const std::vector<Matrix *> &grad_cells_vec);
        virtual RnnContext *init();
        virtual void release(RnnContext *);
        virtual std::vector<Parameters*> get_parameters();
        virtual void zero_grad();
        virtual uint get_hidden_num() {
            return hidden_num;
        }
    private:
        uint input_num;
        uint hidden_num;
        DATATYPE sigma;
        Parameters *wxh;
        Parameters *whh;
        Parameters *bh;
        Parameters *wxi;
        Parameters *whi;
        Parameters *bi;
        Parameters *wxf;
        Parameters *whf;
        Parameters *bf;
        Parameters *wxo;
        Parameters *who;
        Parameters *bo;
        Parameters *wxc;
        Parameters *whc;
        Parameters *bc;
        bool rand;
};

#endif