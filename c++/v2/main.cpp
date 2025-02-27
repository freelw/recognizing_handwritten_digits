#include "layers.h"

#include <iostream>


void test_crossentropyloss() {

    Matrix *Input = allocTmpMatrix(Shape(10, 2));
    for (auto i = 0; i < 2; ++ i) {
        (*Input)[0][i] = 0.1;
        (*Input)[1][i] = 0.1;
        (*Input)[2][i] = 0.1;
        (*Input)[3][i] = 0.1;
        (*Input)[4][i] = 0.1;
        (*Input)[5][i] = 0.1;
        (*Input)[6][i] = 0.1;
        (*Input)[7][i] = 0.1;
        (*Input)[8][i] = 0.15;
        (*Input)[9][i] = 0.05;
    }

    CrossEntropyLoss loss_fn({6, 8});
    Context *ctx = loss_fn.init();
    auto loss = loss_fn.forward(ctx, Input);
    std::cout << "loss : " << *loss << std::endl;
    auto grad = loss_fn.backward(ctx, nullptr);
    std::cout << "grad : " << *grad << std::endl;
    loss_fn.release(ctx);

}
int main() {


    test_crossentropyloss();
    return 0;
}