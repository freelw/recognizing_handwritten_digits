#include "layers.h"

#include <cmath>
#include <vector>



Matrix *CrossEntropyLoss::forward(Context *, Matrix *input) {

    Matrix *mExp = allocTmpMatrix(input);
    for (uint i = 0; i < mExp->getShape().rowCnt; ++ i) {
        DATATYPE max = *mExp[i][0];
        for (uint j = 1; j < input->getShape().colCnt; ++ j) {
            auto & e = *mExp[i][j];
            if (max > e) {
                max = e;
            }
        }

        for (uint j = 1; j < input->getShape().colCnt; ++ j) {
            auto & e = *mExp[i][j];
            e = std::exp(e-max);
        }
        
    }
}

Matrix *CrossEntropyLoss::backward(Context *, Matrix *grad) {

}