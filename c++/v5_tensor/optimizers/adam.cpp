#include "adam.h"

void Adam::step() {

}

void Adam::clip_grad(float grad_clip_val) {
    std::vector<Tensor *> grads;
    grads.reserve(parameters.size());
    for (auto param : parameters) {
        assert(param->is_require_grad());
        assert(param->get_grad() != nullptr);
        grads.push_back(param->get_grad());
    }
    Tensor *norm = allocTensor({1}, "clip_grad_norm");
    gCreateAction(
        new CalcAllGradNormAction(grads, norm)
    );
}