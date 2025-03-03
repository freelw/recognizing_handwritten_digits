#include "models.h"
#include <iostream>
#include "dataloader/mnist_loader_base.h"
#include <algorithm>
#include <random>
#include <chrono>
#include "optimizers.h"

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

void valid_param(DATATYPE v) {
    assert(!std::isnan(v));
}

double update_mini_batch(
    MLP &m,
    std::vector<TrainingData*> &mini_batch,
    Adam &optimizer) {
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
    optimizer.step();
    freeTmpMatrix();
    return loss;
}

int evaluate(MLP &m, std::vector<TrainingData*> &v_test_data) {
    int sum = 0;
    for (uint i = 0; i < v_test_data.size(); ++ i) {
        Matrix *res = m.forward(v_test_data[i]->x);
        res->checkShape(Shape(10, 1));
        uint index = 0;
        for (uint j = 1; j < res->getShape().rowCnt; ++ j) {
            if ((*res)[j][0] > (*res)[index][0]) {
                index = j;
            }
        }
        if (index == v_test_data[i]->y) {
            sum ++;
        }
    }
    return sum;
}

void SGD(MLP &m, std::vector<TrainingData*> &v_training_data,
    std::vector<TrainingData*> &v_test_data,
    int epochs, int mini_batch_size, Adam &optimizer, bool eval) {

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
            loss_sum += update_mini_batch(m, mini_batches[i], optimizer);
        }
        cout << "epoch : [" << e+1 << "/" << epochs << "] loss : " << loss_sum / mini_batches.size() << endl;
        if (eval) {
            std::cout << evaluate(m, v_test_data) << " / " << v_test_data.size() << std::endl;
        }
        freeTmpMatrix();
    }
}

void train(int epochs, int batch_size, bool use_dropout, bool eval) {
    cout << "eval : " << eval << endl;
    MnistLoaderBase loader;
    loader.load();
    std::vector<TrainingData*> v_training_data;
    std::vector<TrainingData*> v_test_data;
    for (auto i = 0; i < TRAIN_IMAGES_NUM; ++ i) {
        TrainingData *p = new TrainingData(INPUT_LAYER_SIZE, loader.getTrainLabels()[i]);
        for (auto j = 0; j < INPUT_LAYER_SIZE; ++ j) {
            // (*(p->x))[j][0] = loader.getTrainImages()[i][j]*1./256;
            (*(p->x))[j][0] = loader.getTrainImages()[i][j];
        }
        v_training_data.emplace_back(p);
    }
    for (auto i = 0; i < TEST_IMAGES_NUM; ++ i) {
        int index = i + TRAIN_IMAGES_NUM;
        TrainingData *p = new TrainingData(INPUT_LAYER_SIZE, loader.getTrainLabels()[index]);
        for (auto j = 0; j < INPUT_LAYER_SIZE; ++ j) {
            //(*(p->x))[j][0] = loader.getTrainImages()[index][j]*1./256;
            (*(p->x))[j][0] = loader.getTrainImages()[index][j];
        }
        v_test_data.emplace_back(p);
    }
    cout << "data loaded." << endl;
    assert(v_training_data.size() == TRAIN_IMAGES_NUM);
    assert(v_test_data.size() == TEST_IMAGES_NUM);

    MLP m(INPUT_LAYER_SIZE, {30, 10});
    m.init();
    Adam adam(m.get_parameters(), 0.001);
    SGD(m, v_training_data, v_test_data, epochs, batch_size, adam, eval);
    for (uint i = 0; i < v_training_data.size(); ++ i) {
        delete v_training_data[i];
    }
    for (uint i = 0; i < v_test_data.size(); ++ i) {
        delete v_test_data[i];
    }
}

void testgrad();

int main(int argc, char *argv[]) {
    bool test = false;
    if (argc == 2) {
        if (std::string(argv[1]) == "test") {
            test = true;
        }
    }
    if (test) {
        testgrad();
    } else {
        if (argc != 5) {
            std::cout << "Usage: " << argv[0] << " <epochs> <batch_size> <use_dropout> <eval>" << std::endl;
            return 1;
        }
        int epochs = atoi(argv[1]);
        int batch_size = atoi(argv[2]);
        int use_dropout = atoi(argv[3]);
        int eval = atoi(argv[4]);
        train(epochs, batch_size, use_dropout == 1, eval == 1);
    }
    return 0;
}