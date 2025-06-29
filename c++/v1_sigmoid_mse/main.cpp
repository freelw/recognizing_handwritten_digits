#include "variable.h"
#include "dataloader/mnist_loader_base.h"
#include "mlp.h"

#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <math.h>

#define INPUT_LAYER_SIZE 784

class TrainingData {
public:
    std::vector<double> x;
    int y;
};

void print_progress(const std::string& prefix, uint i, uint tot) {
    std::cout << "\r" << prefix << " [" << i << "/" << tot << "]" << std::flush;
}

double update_mini_batch(
    int epoch,
    Model& m,
    std::vector<TrainingData*>& mini_batch,
    double eta) {

    VariablePtr loss_sum = allocTmpVar(0);
    for (uint i = 0; i < mini_batch.size(); ++i) {
        std::vector<VariablePtr> input;
        for (uint j = 0; j < INPUT_LAYER_SIZE; ++j) {
            input.emplace_back(allocTmpVar(mini_batch[i]->x[j]));
        }
        std::vector<VariablePtr> res = m.forward(input);
        VariablePtr loss = MSELoss(res, mini_batch[i]->y);
        loss_sum = *loss_sum + loss;
    }

    VariablePtr avg_loss = *loss_sum / allocTmpVar(mini_batch.size());
    m.zeroGrad();
    double ret = avg_loss->getValue();
    avg_loss->setGradient(1);
    avg_loss->bp();
    m.update(eta);
    destroyTmpVars();
    return ret;
}

void evaluate(
    Model& m,
    std::vector<TrainingData*>& v_test_data) {
    int correct = 0;
    for (uint i = 0; i < v_test_data.size(); ++i) {
        std::vector<VariablePtr> input;
        for (auto j = 0; j < INPUT_LAYER_SIZE; ++j) {
            input.emplace_back(allocTmpVar(v_test_data[i]->x[j]));
        }
        std::vector<VariablePtr> res = m.forward(input);
        int max_index = 0;
        double max_value = res[0]->getValue();
        for (uint j = 1; j < res.size(); ++j) {
            if (res[j]->getValue() > max_value) {
                max_value = res[j]->getValue();
                max_index = j;
            }
        }
        if (max_index == v_test_data[i]->y) {
            correct++;
        }
        destroyTmpVars();
    }
    std::cout << "correct: " << correct << " / " << v_test_data.size() << std::endl;
}

void SGD(
    std::vector<TrainingData*>& v_training_data,
    std::vector<TrainingData*>& v_test_data,
    int epochs, int mini_batch_size, double eta) {

    std::vector<uint> sizes;
    sizes.push_back(30);
    sizes.push_back(10);

    Model* m = new Model(INPUT_LAYER_SIZE, sizes);

    int n = v_training_data.size();
    for (auto e = 0; e < epochs; ++e) {
        auto rng = std::default_random_engine{};
        std::shuffle(std::begin(v_training_data), std::end(v_training_data), rng);
        std::vector<std::vector<TrainingData*>> mini_batches;
        for (auto i = 0; i < n; i += mini_batch_size) {
            std::vector<TrainingData*> tmp;
            auto end = std::min(i + mini_batch_size, n);
            tmp.assign(v_training_data.begin() + i, v_training_data.begin() + end);
            mini_batches.emplace_back(tmp);
        }
        double loss_sum = 0;
        for (uint i = 0; i < mini_batches.size(); ++i) {
            loss_sum += update_mini_batch(e, *m, mini_batches[i], eta);
            print_progress("Processing mini-batch", i, mini_batches.size());
        }
        std::cout << "epoch : [" << e + 1 << "/" << epochs << "] loss : " << loss_sum / mini_batches.size() << std::endl;
        evaluate(*m, v_test_data);
    }
}

void loadData(
    std::vector<TrainingData*>& v_training_data,
    std::vector<TrainingData*>& v_test_data) {
    MnistLoaderBase loader;
    loader.load();
    for (auto i = 0; i < TRAIN_IMAGES_NUM; ++i) {
        TrainingData* p = new TrainingData();
        for (auto j = 0; j < INPUT_LAYER_SIZE; ++j) {
            p->x.emplace_back(loader.getTrainImages()[i][j] * 1. / 256);
            p->y = loader.getTrainLabels()[i];
        }
        v_training_data.emplace_back(p);
    }
    for (auto i = 0; i < TEST_IMAGES_NUM; ++i) {
        int index = i + TRAIN_IMAGES_NUM;
        TrainingData* p = new TrainingData();
        for (auto j = 0; j < INPUT_LAYER_SIZE; ++j) {
            p->x.emplace_back(loader.getTrainImages()[index][j] * 1. / 256);
            p->y = loader.getTrainLabels()[index];
        }
        v_test_data.emplace_back(p);
    }
}

void train(int epochs, int batch_size) {
    std::vector<TrainingData*> v_training_data;
    std::vector<TrainingData*> v_test_data;
    loadData(v_training_data, v_test_data);
    std::cout << "data loaded." << std::endl;
    SGD(v_training_data, v_test_data, epochs, batch_size, 0.01);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <epochs> <batch_size>" << std::endl;
        return 1;
    }
    int epochs = atoi(argv[1]);
    int batch_size = atoi(argv[2]);
    train(epochs, batch_size);
    return 0;
}