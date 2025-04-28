#include <iostream>
#include "autograd/node.cuh"
#include "autograd/optimizers.cuh"
#include <iostream>
#include "dataloader/mnist_loader_base.h"
#include "mlp.cuh"
#include <algorithm>
#include <random>
#include <chrono>
#include "backends/cpu/cpu_ops.cuh"
#include "backends/gpu/cuda_ops.cuh"

#define INPUT_LAYER_SIZE 784

Matrix *allocMatrix(Shape shape);

DATATYPE update_mini_batch(
    autograd_cuda::MLP &m,
    std::vector<TrainingData*> &mini_batch,
    autograd_cuda::Adam &optimizer) {
    Matrix *input = allocTmpMatrix(Shape(INPUT_LAYER_SIZE, mini_batch.size()));
    std::vector<uint> labels;
    for (uint i = 0; i < INPUT_LAYER_SIZE; ++ i) {
        for (uint j = 0; j < mini_batch.size(); ++ j) {
            //(*input)[i][j] = mini_batch[j]->x[i];
            input->set_val(i, j, mini_batch[j]->x[i]);
        }
    }
    input->increase_cpu_ver();
    input->sync();
    labels.reserve(mini_batch.size());
    for (uint j = 0; j < mini_batch.size(); ++ j) {
        labels.emplace_back(mini_batch[j]->y);
    }
    optimizer.zero_grad();
    auto loss = m.forward(autograd_cuda::allocNode(input))->CrossEntropy(labels);
    assert(loss->get_weight()->getShape().rowCnt == 1);
    assert(loss->get_weight()->getShape().colCnt == 1);
    loss->get_weight()->sync();
    DATATYPE ret = loss->get_weight()->get_val(0, 0);
    loss->backward();
    optimizer.step();
    autograd_cuda::freeAllNodes();
    autograd_cuda::freeAllEdges();
    freeTmpMatrix();
    return ret;
}

int evaluate(autograd_cuda::MLP &m, std::vector<TrainingData*> &v_test_data) {
    int sum = 0;
    for (uint i = 0; i < v_test_data.size(); ++ i) {
        Matrix *input = allocTmpMatrix(v_test_data[i]->x);
        Matrix *res = m.forward(autograd_cuda::allocNode(input))->get_weight();
        res->sync();
        res->checkShape(Shape(10, 1));
        uint index = res->argMax()[0];
        if (index == v_test_data[i]->y) {
            sum ++;
        }
    }
    return sum;
}

void fit(autograd_cuda::MLP &m, std::vector<TrainingData*> &v_training_data,
    std::vector<TrainingData*> &v_test_data,
    int epochs, int mini_batch_size, autograd_cuda::Adam &optimizer, bool eval) {

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
        autograd_cuda::freeAllNodes();
        autograd_cuda::freeAllEdges();
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
            p->x.emplace_back(loader.getTrainImages()[i][j]*1./256);
        }
        v_training_data.emplace_back(p);
    }
    for (auto i = 0; i < TEST_IMAGES_NUM; ++ i) {
        int index = i + TRAIN_IMAGES_NUM;
        TrainingData *p = new TrainingData(INPUT_LAYER_SIZE, loader.getTrainLabels()[index]);
        for (auto j = 0; j < INPUT_LAYER_SIZE; ++ j) {
            p->x.emplace_back(loader.getTrainImages()[index][j]*1./256);
        }
        v_test_data.emplace_back(p);
    }
    cout << "data loaded." << endl;
    
    autograd_cuda::MLP m(INPUT_LAYER_SIZE, {30, 10});
    autograd_cuda::Adam adam(m.get_parameters(), 0.001);
    fit(m, v_training_data, v_test_data, epochs, batch_size, adam, eval);
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
    g_backend_ops = new CPUBackendOps();
    g_gpu_backend_ops = new GPUBackendOps();
    train(epochs, batch_size, use_dropout == 1, eval == 1);
    freeTmpMatrix();
    return 0;
}