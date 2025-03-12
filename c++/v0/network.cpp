#include "network.h"
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>

NetWork::NetWork(const std::vector<int> &_sizes)
        : sizes(_sizes), num_layers(_sizes.size()) {
    for (uint i = 1; i < sizes.size(); ++ i) {
        biases.emplace_back((new Matrix(Shape(sizes[i], 1)))->zero());
    }
    for (uint i = 1; i < sizes.size(); ++ i) {
        weights.emplace_back((new Matrix(Shape(sizes[i], sizes[i-1])))->zero());
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<DATATYPE> distribution(0.0, 1.0);
    
    const int L = sizes.size() - 1;
    for (auto i = 0; i < L; ++ i) {
        Shape bs = biases[i]->getShape();
        for (uint j = 0; j < bs.rowCnt; ++ j) {
            for (uint k = 0; k < bs.colCnt; ++ k) {
                // assert(biases[i].valid(j, k));
                (*biases[i])[j][k] = distribution(generator);
            }
        }
    }

    for (auto i = 0; i < L; ++ i) {
        Shape ws = weights[i]->getShape();
        for (uint j = 0; j < ws.rowCnt; ++ j) {
            for (uint k = 0; k < ws.colCnt; ++ k) {
                // assert(weights[i]->valid(j, k));
                (*weights[i])[j][k] = distribution(generator);
            }
        }
    }

    assert(biases.size() == weights.size());
}

Matrix *NetWork::feedforward(Matrix *a) {
    Matrix *res = allocTmpMatrix(a);
    for (uint i = 0; i < sizes.size()-1; ++ i) {
        res = (*(weights[i]->at(*res))+ *biases[i])->sigmoid();
    }
    return res;
}

void NetWork::SGD(
    std::vector<TrainingData*> &v_training_data,
    std::vector<TrainingData*> &v_test_data,
    int epochs, int mini_batch_size, DATATYPE eta, bool eval) {

    int n = v_training_data.size();
    for (auto e = 0; e < epochs; ++ e) {
        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(v_training_data), std::end(v_training_data), rng);
        std::vector<std::vector<TrainingData*>> mini_batches;
        for (auto i = 0; i < n; i += mini_batch_size) {
            std::vector<TrainingData*> tmp;
            auto end = min(i+mini_batch_size, n);
            tmp.assign(v_training_data.begin()+i,v_training_data.begin()+end);
            mini_batches.emplace_back(tmp);
        }

        for (uint i = 0; i < mini_batches.size(); ++ i) {
            update_mini_batch(mini_batches[i], eta);
        }

        std::cout << "Epoch " << e;
        if (eval) {
            std::cout << " : " << evaluate(v_test_data) << " / " << v_test_data.size();
        }
        std::cout << std::endl;
    }
    std::cout << "final eval : " << evaluate(v_test_data) << " / " << v_test_data.size() << std::endl;
}

void NetWork::update_mini_batch(
    std::vector<TrainingData*> &mini_batch,
    DATATYPE eta) {
        
    std::vector<Matrix*> nabla_b;
    std::vector<Matrix*> nabla_w;
    const auto L = sizes.size() - 1;
    for (uint i = 0; i < L; ++ i) {
        nabla_b.emplace_back(allocTmpMatrix(biases[i]->getShape()));
    }

    for (uint i = 0; i < L; ++ i) {
        nabla_w.emplace_back(allocTmpMatrix(weights[i]->getShape()));
    }

    for (uint i = 0; i < mini_batch.size(); ++ i) {
        std::vector<Matrix*> delta_nabla_b;
        std::vector<Matrix*> delta_nabla_w;
        Matrix *y = allocTmpMatrix(Shape(sizes[L], 1));
        (*y)[mini_batch[i]->y][0] = 1;
        backprop(mini_batch[i]->x, y, delta_nabla_b, delta_nabla_w);
        for (uint j = 0; j < L; ++ j) {
            nabla_b[j] = *nabla_b[j] + *delta_nabla_b[j];
            nabla_w[j] = *nabla_w[j] + *delta_nabla_w[j];
        }
    }

    for (uint i = 0; i < L; ++ i) {
        weights[i]->assign(*(weights[i]) - *(*(*nabla_w[i] * eta) / mini_batch.size()));
        biases[i]->assign(*(biases[i]) - *(*(*nabla_b[i] * eta) / mini_batch.size()));
    }

    freeTmpMatrix();
}

void NetWork::backprop(
    Matrix *x, Matrix *y,
    std::vector<Matrix*> &delta_nabla_b,
    std::vector<Matrix*> &delta_nabla_w) {
    
    const auto L = sizes.size() - 1;
    for (uint i = 0; i < L; ++ i) {
        delta_nabla_b.emplace_back(allocTmpMatrix(biases[i]->getShape()));
    }
    for (uint i = 0; i < L; ++ i) {
        delta_nabla_w.emplace_back(allocTmpMatrix(weights[i]->getShape()));
    }

    Matrix *activation = allocTmpMatrix(x);
    std::vector<Matrix*> activations;
    activations.emplace_back(activation);
    std::vector<Matrix*> zs;
    for (uint i = 0; i < L; ++ i) {
        Matrix *z = *(weights[i]->at(*activation)) + *biases[i];
        zs.emplace_back(z);
        activation = z->sigmoid();
        activations.emplace_back(activation);
    }
    assert(activations.size() == L + 1);
    Matrix *delta = *cost_derivative(activations[L], y) * *(zs[L-1]->sigmoid_prime());
    for (int l = L-1; l >= 0; -- l) {
        delta_nabla_b[l] = delta;
        auto activation_transpose = activations[l]->transpose();
        delta_nabla_w[l] = delta->at(*activation_transpose);
        if (l >= 1) {
            delta = *(weights[l]->transpose()->at(*delta)) * *(zs[l-1]->sigmoid_prime());
        }
    }
}

int NetWork::evaluate(std::vector<TrainingData*> &v_test_data) {
    int sum = 0;
    for (uint i = 0; i < v_test_data.size(); ++ i) {
        Matrix *res = feedforward(v_test_data[i]->x);
        int index = 0;
        for (int j = 1; j < sizes[sizes.size() - 1]; ++ j) {
            // assert(res.valid(j, 0) && res.valid(index, 0));
            if ((*res)[j][0] > (*res)[index][0]) {
                index = j;
            }
        }
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wsign-compare"    
        if (index == v_test_data[i]->y) {
            sum ++;
        }
        #pragma GCC diagnostic pop
    }
    return sum;
}

Matrix *NetWork::cost_derivative(
    Matrix *output_activations,
    Matrix *y) {
    return *output_activations - *y;
}

ostream &operator<<(ostream &output, NetWork &s) {
    const int L = s.sizes.size() - 1;
    output << "biases : " << endl;
    for (auto i = 0; i < L; ++ i) {
        output << s.biases[i] << endl;
    }
    output << "weights : " << endl;
    for (auto i = 0; i < L; ++ i) {
        output << s.weights[i] << endl;
    }
    return output;
}