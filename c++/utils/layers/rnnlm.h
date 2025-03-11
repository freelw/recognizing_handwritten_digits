#include "layers/layers.h"

class RnnLMContext {
    public:
        RnnContext *rnn_ctx;
        LinerContext *fc_ctx;
};

class RnnLM {
    public:
        RnnLM(RnnBase *_rnn, uint vocab_size, bool rand);
        virtual ~RnnLM();
        virtual Matrix *forward(RnnLMContext *, const std::vector<Matrix*> &inputs);
        virtual void backward(RnnLMContext *, Matrix* grad);
        virtual RnnLMContext *init();
        virtual void release(RnnLMContext *);
        virtual std::vector<Parameters*> get_parameters();
        std::string predict(const std::string &prefix, uint num_preds);
    private:
        RnnBase *rnn;
        uint vocab_size;
        Liner *fc;
};