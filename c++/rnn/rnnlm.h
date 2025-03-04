#include "layers/layers.h"

class RnnLMContext {
    public:
        RnnContext *rnn_ctx;
        LinerContext *fc_ctx;
};

class RnnLM {
    public:
        RnnLM(Rnn *_rnn, uint vocab_size);
        virtual ~RnnLM();
        virtual Matrix *forward(RnnLMContext *, const std::vector<Matrix*> &inputs);
        virtual Matrix *backward(RnnLMContext *, Matrix* grad);
        virtual RnnLMContext *init();
        virtual void release(RnnContext *);
        virtual std::vector<Parameters*> get_parameters();
        virtual void zero_grad();
    private:
        Rnn *rnn;
        uint vocab_size;
        Liner *fc;
};