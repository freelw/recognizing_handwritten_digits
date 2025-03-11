#include <iostream>
#define OMP_THREADS 4
#include "matrix/matrix.h"
#include <random>
#include <chrono>

void init_weight(Matrix *weight, DATATYPE sigma) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::normal_distribution<DATATYPE> distribution_w(0.0, sigma);
    for (uint i = 0; i < weight->getShape().rowCnt; ++ i) {
        for (uint j = 0; j < weight->getShape().colCnt; ++ j) {
            (*weight)[i][j] = distribution_w(generator_w);
        }
    }
}

int main() {
    uint hidden_size = 32;
    uint input_size = 28;
    uint batch_size = 1024;
    uint output_size = 28;
    uint num_steps = 32;
    Matrix *Input = new Matrix(Shape(input_size, batch_size));
    init_weight(Input, 0.01);
    Matrix *Wx = new Matrix(Shape(hidden_size, input_size));
    Matrix *Wh = new Matrix(Shape(hidden_size, hidden_size));
    Matrix *B = new Matrix(Shape(hidden_size, 1));
    Matrix *Wo = new Matrix(Shape(output_size, hidden_size));
    Matrix *Bo = new Matrix(Shape(output_size, 1));

    cout << "OMP_THREADS : " << OMP_THREADS << endl;
    for (auto epoch = 0; epoch < 1; ++ epoch) {
        for (auto i = 0; i < 170; ++ i) {
            Matrix *H = allocTmpMatrix(Shape(hidden_size, batch_size));
            for (uint j = 0; j < num_steps; ++ j) {
                auto z = (*(Wx->at(*Input)) + *(Wh->at(*H)))->expand_add(*B);
                H = z->tanh();
                auto o = *(Wo->at(*H))->expand_add(*Bo);
            }
            cout << "\repoch : " << epoch << " i : " << i << flush;
            freeTmpMatrix();
        }
        cout << endl;
    }
    delete Input;
    delete Wx;
    delete Wh;
    delete B;
    delete Wo;
    delete Bo;
    return 0;
}