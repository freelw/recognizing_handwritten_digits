#include <iostream>
#define OMP_THREADS 8
#include "matrix/matrix.h"
#include <random>
#include <chrono>

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