#include <iostream>
#include "autograd/node.h"
#include "autograd/optimizers.h"
#include <iostream>
#include "dataloader/mnist_loader_base.h"
#include "mlp.h"
#include <algorithm>
#include <random>
#include <chrono>

#define INPUT_LAYER_SIZE 784

Matrix *allocMatrix(Shape shape);

void testgrad() {
    Matrix *mW1 = allocMatrix(Shape(4, 7))->fill(0.1);
    Matrix *mb1 = allocMatrix(Shape(4, 1))->fill(0.1);
    Matrix *mW2 = allocMatrix(Shape(3, 4))->fill(0.1);
    Matrix *mb2 = allocMatrix(Shape(3, 1))->fill(0.1);
    Matrix *mX = allocMatrix(Shape(7, 30));

    (*mW1)[0][0] = 0.9;
    (*mW1)[1][0] = -0.9;
    (*mW2)[0][0] = 0.9;
    (*mW2)[1][0] = -0.9;

    std::cout << *mW1 << std::endl;
    std::cout << *mW2 << std::endl;

    std::vector<uint> labels;
    for (uint j = 0; j < 15; ++ j) {
        for (uint i = 0; i < 7; ++ i) {
            (*mX)[i][j*2] = 10 + i;
            (*mX)[i][j*2+1] = 10 - i;
        }
        labels.push_back(1);
        labels.push_back(0);
    }

    std::vector<autograd::Parameters *> parameters;
    auto W1 = autograd::allocNode(mW1);
    auto b1 = autograd::allocNode(mb1);
    auto W2 = autograd::allocNode(mW2);
    auto b2 = autograd::allocNode(mb2);
    W1->require_grad();
    b1->require_grad();
    W2->require_grad();
    b2->require_grad();
    auto X = autograd::allocNode(mX);
    parameters.push_back(new autograd::Parameters(W1));
    parameters.push_back(new autograd::Parameters(b1));
    parameters.push_back(new autograd::Parameters(W2));
    parameters.push_back(new autograd::Parameters(b2));

    autograd::Adam adam(parameters, 0.001);
    //auto Z1 = X->at(W1)->expand_add(b1)->Relu();
    // std::cout << "X : " << *X->get_weight() << std::endl;
    // std::cout << "W1 : " << *W1->get_weight() << std::endl;
    // std::cout << "b1 : " << *b1->get_weight() << std::endl;
    for (auto k = 0; k < 20; ++ k) {
        auto Z1 = W1->at(X)->expand_add(b1)->Relu();
        assert(Z1->get_weight()->getShape().rowCnt == 4);
        assert(Z1->get_weight()->getShape().colCnt == 30);
        // std::cout << "Z1 : " << *(Z1->get_weight()) << std::endl;
        
        // auto Z2 = Z1->at(W2)->expand_add(b2)->Relu();
        auto Z2 = W2->at(Z1)->expand_add(b2);
        assert(Z2->get_weight()->getShape().rowCnt == 3);
        assert(Z2->get_weight()->getShape().colCnt == 30);

        // std::cout << "Z2 : " << *(Z2->get_weight()) << std::endl;
        auto loss = Z2->CrossEntropy(labels);
        assert(loss->get_weight()->getShape().rowCnt == 1);
        assert(loss->get_weight()->getShape().colCnt == 1);
    // std::cout << "loss : " << *(loss->get_weight()) << std::endl;
        adam.zero_grad();
        loss->backward();
        adam.step();

        if (k == 19) {
            std::cout << *(W1->get_weight()) << std::endl;
            std::cout << *(W1->get_grad()) << std::endl;
            std::cout << *(b1->get_weight()) << std::endl;
            std::cout << *(b1->get_grad()) << std::endl;
            std::cout << *(W2->get_weight()) << std::endl;
            std::cout << *(W2->get_grad()) << std::endl;
            std::cout << *(b2->get_weight()) << std::endl;
            std::cout << *(b2->get_grad()) << std::endl;
        }
        freeTmpMatrix();
    }

    autograd::freeAllNodes();
    autograd::freeAllEdges();
    freeTmpMatrix();

    for (auto p : parameters) {
        delete p;
    }
    delete mW1;
    delete mb1;
    delete mW2;
    delete mb2;
    delete mX;
}

DATATYPE update_mini_batch(
    autograd::MLP &m,
    std::vector<TrainingData*> &mini_batch,
    autograd::Adam &optimizer) {
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
    optimizer.zero_grad();
    auto loss = m.forward(autograd::allocNode(input))->CrossEntropy(labels);
    assert(loss->get_weight()->getShape().rowCnt == 1);
    assert(loss->get_weight()->getShape().colCnt == 1);
    DATATYPE ret = *(loss->get_weight())[0][0];
    loss->backward();
    optimizer.step();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
    freeTmpMatrix();
    return ret;
}

int evaluate(autograd::MLP &m, std::vector<TrainingData*> &v_test_data) {
    int sum = 0;
    for (uint i = 0; i < v_test_data.size(); ++ i) {
        Matrix *res = m.forward(autograd::allocNode(v_test_data[i]->x))->get_weight();
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

void SGD(autograd::MLP &m, std::vector<TrainingData*> &v_training_data,
    std::vector<TrainingData*> &v_test_data,
    int epochs, int mini_batch_size, autograd::Adam &optimizer, bool eval) {

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
        autograd::freeAllNodes();
        autograd::freeAllEdges();
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
    
    autograd::MLP m(INPUT_LAYER_SIZE, {30, 10});
    autograd::Adam adam(m.get_parameters(), 0.001);
    SGD(m, v_training_data, v_test_data, epochs, batch_size, adam, eval);
    for (uint i = 0; i < v_training_data.size(); ++ i) {
        delete v_training_data[i];
    }
    for (uint i = 0; i < v_test_data.size(); ++ i) {
        delete v_test_data[i];
    }
}

int main(int argc, char *argv[]) {
    // testgrad();
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <epochs> <batch_size> <use_dropout> <eval>" << std::endl;
        return 1;
    }
    int epochs = atoi(argv[1]);
    int batch_size = atoi(argv[2]);
    int use_dropout = atoi(argv[3]);
    int eval = atoi(argv[4]);
    train(epochs, batch_size, use_dropout == 1, eval == 1);
    freeTmpMatrix();
    return 0;
}