#include "matrix/matrix.h"
#include "network.h"
#include "dataloader/mnist_loader_base.h"
#include <iostream>
#include <vector>
#include <assert.h>

#define INPUT_LAYER_SIZE 784
using namespace std;

void train(bool eval) {
    cout << "eval : " << eval << endl;

    vector<int> sizes;
    sizes.push_back(INPUT_LAYER_SIZE);
    sizes.push_back(30);
    sizes.push_back(10);
    NetWork mynet(sizes);
    MnistLoaderBase loader;
    loader.load();
    
    std::vector<TrainingData*> v_training_data;
    std::vector<TrainingData*> v_test_data;
    for (auto i = 0; i < TRAIN_IMAGES_NUM; ++ i) {
        TrainingData *p = new TrainingData(INPUT_LAYER_SIZE, loader.getTrainLabels()[i]);
        for (auto j = 0; j < INPUT_LAYER_SIZE; ++ j) {
            p->x[j][0] = loader.getTrainImages()[i][j]*1./256;
        }
        v_training_data.emplace_back(p);
    }
    for (auto i = 0; i < TEST_IMAGES_NUM; ++ i) {
        int index = i + TRAIN_IMAGES_NUM;
        TrainingData *p = new TrainingData(INPUT_LAYER_SIZE, loader.getTrainLabels()[index]);
        for (auto j = 0; j < INPUT_LAYER_SIZE; ++ j) {
            p->x[j][0] = loader.getTrainImages()[index][j]*1./256;
        }
        v_test_data.emplace_back(p);
    }
    cout << "data loaded." << endl;

    assert(v_training_data.size() == TRAIN_IMAGES_NUM);
    assert(v_test_data.size() == TEST_IMAGES_NUM);
    mynet.SGD(v_training_data, v_test_data, 30, 30, 3, eval);

    for (uint i = 0; i < v_training_data.size(); ++ i) {
        delete v_training_data[i];
    }
    for (uint i = 0; i < v_test_data.size(); ++ i) {
        delete v_test_data[i];
    }
}

void test() {


    const int x = 5;
    const int y = 6;
    const int z = 7;
    Shape s0(y, x);
    Shape s1(z, y);
    Shape s3(z, 1);
    Shape si(x, 1);

    Matrix a(s0), b(s1), bias(s3), i(si);

    cout << i << endl;
    cout << a << endl;
    cout << b << endl;
    auto c = b.dot(a.dot(i));
    cout << c << endl;
    auto d = c + bias;

    auto e = sigmoid(d);

    cout << e << endl;
}
int main(int argc, char *argv[])
{
    bool eval = false;

    if (argc == 2) {
        if (std::string(argv[1]) == "eval") {
            eval = true;
        }
    }

    test();
    train(eval);
    
    return 0;
}