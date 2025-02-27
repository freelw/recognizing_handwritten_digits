#include "layers.h"

#include <cmath>
#include <vector>
#include <assert.h>

Matrix *Relu::forward(Context *ctx, Matrix *input) {
    ReluContext *rl_ctx = (ReluContext*)ctx;
    rl_ctx->input = input;
    auto shape = input->getShape();
    Matrix *res = allocTmpMatrix(shape);
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            auto &value = (*input)[i][j];
            (*res)[i][j] = value > 0 ? value : 0;
        }
    }
    return res;
}

Matrix *Relu::backward(Context *ctx, Matrix *grad) {
    ReluContext *rl_ctx = (ReluContext*)ctx;
    auto &input = rl_ctx->input;
    input->checkShape(*grad);
    Matrix *res_grad = allocTmpMatrix(grad->getShape());
    auto shape = grad->getShape();
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            auto &value = (*input)[i][j];
            (*res_grad)[i][j] = value > 0 ? (*grad)[i][j]*value : 0;
        }
    }
    return res_grad;
}

Context *Relu::init() {
    return new ReluContext();
}

void Relu::release(Context * ctx) {
    ReluContext *rl_ctx = (ReluContext*)ctx;
    delete rl_ctx;
}

Matrix *CrossEntropyLoss::forward(Context * ctx, Matrix *input) {
    CrossEntropyLossContext *ce_ctx = (CrossEntropyLossContext*)ctx;
    ce_ctx->input = input;
    assert(input->getShape().colCnt == labels.size());
    Matrix *mExp = allocTmpMatrix(input);
    Matrix *ce_for_eachbach = allocTmpMatrix(Shape(1, labels.size()));
    Matrix *loss = allocTmpMatrix(Shape(1,1));
    DATATYPE loss_value = 0;
    for (uint j = 0; j < input->getShape().colCnt; ++ j) {
        DATATYPE max = (*mExp)[0][j];
        for (uint i = 0; i < mExp->getShape().rowCnt; ++ i) {
            auto & e = (*mExp)[i][j];
            if (max > e) {
                max = e;
            }
        }
        DATATYPE sum = 0;
        for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
            auto & e = (*mExp)[i][j];
            e = std::exp(e-max);
            sum += e;
        }
        auto target = labels[j];
        auto ez = (*mExp)[target][j];
        CrosEntropyInfo p;
        p.ez = ez;
        p.sum = sum;
        p.max = max;
        ce_ctx->info.push_back(p);
        (*ce_for_eachbach)[0][j] = -log(ez/sum);
        loss_value += (*ce_for_eachbach)[0][j];
    }
    (*loss)[0][0] = loss_value/labels.size();
    return loss;
}

Matrix *CrossEntropyLoss::backward(Context *ctx, Matrix *) {
    CrossEntropyLossContext *ce_ctx = (CrossEntropyLossContext*)ctx;
    assert(ce_ctx->info.size() == labels.size());
    Matrix *grad = allocTmpMatrix(Shape(ce_ctx->input->getShape()));
    auto batch_size = labels.size();
    for (uint i = 0; i < batch_size; ++ i) {
        DATATYPE ez = ce_ctx->info[i].ez;
        DATATYPE sum = ce_ctx->info[i].sum;
        DATATYPE max = ce_ctx->info[i].max;
        for (uint j = 0; j < ce_ctx->input->getShape().rowCnt; ++j) {
            (*grad)[j][i] = std::exp((*ce_ctx->input)[j][i] - max) / sum / batch_size;
        }
        auto target = labels[i];
        (*grad)[target][i] -= std::exp((*ce_ctx->input)[target][i] - max) / ez / batch_size;
    }
    return grad;
}

Context *CrossEntropyLoss::init() {
    return new CrossEntropyLossContext();
}

void CrossEntropyLoss::release(Context *ctx) {
    CrossEntropyLossContext *ce_ctx = (CrossEntropyLossContext*)ctx;
    delete ce_ctx;
}