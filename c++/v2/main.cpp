#include "models.h"
#include <iostream>
#include "dataloader/mnist_loader_base.h"
#include <algorithm>
#include <random>
#include <chrono>

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

#define INPUT_LAYER_SIZE 784
using namespace std;

void optimize(const std::vector<Parameters*> &parameters, DATATYPE lr) {

/*
    m = beta1 * m + (1 - beta1) * gradient;
    v = beta2 * v + (1 - beta2) * gradient * gradient;
    double m_hat = m / (1 - std::pow(beta1, t));
    double v_hat = v / (1 - std::pow(beta2, t));
    value -= lr * (m_hat / (std::sqrt(v_hat) + epsilon));
*/

    const DATATYPE beta1 = 0.9;
    const DATATYPE beta2 = 0.95;
    const DATATYPE epsilon = 1e-8;

    for (auto p : parameters) {
        p->inc_t();
        auto t = p->get_t();
        Matrix *weight = p->get_weight();
        Matrix *grad = p->get_grad();
        Matrix *mm = p->get_m();
        Matrix *mv = p->get_v();
        Shape shape = weight->getShape();

        grad->checkShape(shape);
        mm->checkShape(shape);
        mv->checkShape(shape);
        
        for (uint i = 0; i < shape.rowCnt; ++ i) {
            for (uint j = 0; j < shape.colCnt; ++ j) {
                auto &m = (*mm)[i][j];
                auto &v = (*mv)[i][j];
                auto &gradient = (*grad)[i][j];
                auto &value = (*weight)[i][j];
                m = beta1 * m + (1 - beta1) * gradient;
                v = beta2 * v + (1 - beta2) * gradient * gradient;
                double m_hat = m / (1 - std::pow(beta1, t));
                double v_hat = v / (1 - std::pow(beta2, t));
                value -=  lr * (m_hat / (std::sqrt(v_hat) + epsilon));
            }
        }
    }
}

double update_mini_batch(
    MLP &m,
    std::vector<TrainingData*> &mini_batch,
    DATATYPE eta) {
    Matrix *input = allocTmpMatrix(Shape(INPUT_LAYER_SIZE, mini_batch.size()));
    std::vector<uint> labels;
    for (uint i = 0; i < INPUT_LAYER_SIZE; ++ i) {
        for (uint j = 0; j < mini_batch.size(); ++ j) {
            (*input)[i][j] = (*(mini_batch[j]->x))[i][0];
        }
    }
    labels.reserve(mini_batch.size());
    for (uint j = 0; j < mini_batch.size(); ++ j) {
        labels.emplace_back(mini_batch[j]->y);
    }
    m.zero_grad();
    double loss = m.backward(input, labels);
    optimize(m.get_parameters(), eta);
    freeTmpMatrix();
    return loss;
}

void SGD(MLP &m, std::vector<TrainingData*> &v_training_data,
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
        double loss_sum = 0;
        for (uint i = 0; i < mini_batches.size(); ++ i) {
            loss_sum += update_mini_batch(m, mini_batches[i], eta);
        }
        cout << "epoch : [" << e+1 << "/" << epochs << "] loss : " << loss_sum / mini_batches.size() <<  endl; 
    }
}

void train(bool eval) {
    cout << "eval : " << eval << endl;

    MnistLoaderBase loader;
    loader.load();
    
    std::vector<TrainingData*> v_training_data;
    std::vector<TrainingData*> v_test_data;
    for (auto i = 0; i < TRAIN_IMAGES_NUM; ++ i) {
        TrainingData *p = new TrainingData(INPUT_LAYER_SIZE, loader.getTrainLabels()[i]);
        for (auto j = 0; j < INPUT_LAYER_SIZE; ++ j) {
            (*(p->x))[j][0] = loader.getTrainImages()[i][j]*1./256;
        }
        v_training_data.emplace_back(p);
    }
    for (auto i = 0; i < TEST_IMAGES_NUM; ++ i) {
        int index = i + TRAIN_IMAGES_NUM;
        TrainingData *p = new TrainingData(INPUT_LAYER_SIZE, loader.getTrainLabels()[index]);
        for (auto j = 0; j < INPUT_LAYER_SIZE; ++ j) {
            (*(p->x))[j][0] = loader.getTrainImages()[index][j]*1./256;
        }
        v_test_data.emplace_back(p);
    }
    cout << "data loaded." << endl;

    assert(v_training_data.size() == TRAIN_IMAGES_NUM);
    assert(v_test_data.size() == TEST_IMAGES_NUM);

    MLP m(INPUT_LAYER_SIZE, {30, 10});

    SGD(m, v_training_data, v_test_data, 30, 30, 3, eval);
    

    for (uint i = 0; i < v_training_data.size(); ++ i) {
        delete v_training_data[i];
    }
    for (uint i = 0; i < v_test_data.size(); ++ i) {
        delete v_test_data[i];
    }
}

int main() {
    // test_crossentropyloss();

    train(true);


    return 0;
}