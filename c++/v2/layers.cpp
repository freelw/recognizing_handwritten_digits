#include "layers.h"

#include <cmath>
#include <vector>
#include <assert.h>

Matrix *CrossEntropyLoss::forward(Context *, Matrix *input) {
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
        (*ce_for_eachbach)[0][j] = -log(ez/sum);
        loss_value += (*ce_for_eachbach)[0][j];
    }
    (*loss)[0][0] = loss_value/labels.size();
    return loss;
}

Matrix *CrossEntropyLoss::backward(Context *, Matrix *grad) {
    assert(false);
    return nullptr;
}