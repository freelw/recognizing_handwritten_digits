#include "variable.h"
#include "mnist_loader_base.h"
#include "mlp.h"

#include <iostream>

void test0() {
    VariablePtr x = new Parameter(1.0);
    VariablePtr y = new Parameter(2.0);
    VariablePtr z = *((*x) * y) + x;
    std::cout << *z << std::endl;
    z = z->Relu();
    std::cout << *z << std::endl;
    z = z->log();
    std::cout << *z << std::endl;
    z = z->exp();
    std::cout << *z << std::endl;
    z->setGradient(1.0);
    z->bp();
    std::cout << *x << std::endl;
    std::cout << *y << std::endl;
}

#define INPUT_LAYER_SIZE 784

class TrainingData {
public:
    std::vector<double> x;
    int y;
};

int main() {
    MnistLoaderBase loader;
    loader.load();

    std::vector<TrainingData*> v_training_data;
    std::vector<TrainingData*> v_test_data;
    for (auto i = 0; i < TRAIN_IMAGES_NUM; ++ i) {
        TrainingData *p = new TrainingData();
        for (auto j = 0; j < INPUT_LAYER_SIZE; ++ j) {
            p->x.emplace_back(loader.getTrainImages()[i][j]*1./256);
            p->y = loader.getTrainLabels()[i];
        }
        v_training_data.emplace_back(p);
    }
    for (auto i = 0; i < TEST_IMAGES_NUM; ++ i) {
        int index = i + TRAIN_IMAGES_NUM;
        TrainingData *p = new TrainingData();
        for(auto j = 0; j < INPUT_LAYER_SIZE; ++ j) {
            p->x.emplace_back(loader.getTrainImages()[index][j]*1./256);
            p->y = loader.getTrainLabels()[index];
        }
        v_test_data.emplace_back(p);
    }

    std::cout << "data loaded." << std::endl;
    std::vector<int> sizes;
    sizes.push_back(30);
    sizes.push_back(10);
    Model m(INPUT_LAYER_SIZE, sizes);
    for (auto epoch = 0; epoch < 100; ++ epoch) {
        std::cout << "start epoch " << epoch << std::endl;
        VariablePtr loss_sum = allocTmpVar(0);
        for (auto i = 0; i < 10; ++ i) {
            std::vector<VariablePtr> input;    
            for (auto j = 0; j < INPUT_LAYER_SIZE; ++ j) {
                input.emplace_back(allocTmpVar(loader.getTrainImages()[i][j]*1./256));
            }
            std::vector<VariablePtr> res = m.forward(input);
            VariablePtr loss = CrossEntropyLoss(res, loader.getTrainLabels()[i]);
            loss_sum = *loss_sum + loss;
        }
        VariablePtr avg_loss = *loss_sum / allocTmpVar(10);
        avg_loss->setGradient(1);
        avg_loss->bp();
        m.update(0.1);
        destroyTmpVars();
    }
    return 0;
}