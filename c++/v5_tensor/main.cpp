#include <iostream>
#include <unistd.h>
#include "common.h"
#include "tensor.h"
#include "backends/cpu/cpu_ops.h"
#include "graph/node.h"
#include "dataloader/mnist_loader_base.h"
#include "optimizers/adam.h"
#include "model/mlp.h"

void train(int epochs, float lr, int batch_size) {
    // MnistLoaderBase loader;
    // loader.load();
    // std::cout << "data loaded." << std::endl;

    MLP m(784, {30, 10});
    Tensor *inputs = allocTensor({batch_size, 784}, "inputs");
    Tensor *labels = allocTensor({batch_size}, "labels", INT32);
    auto n_inputs = graph::allocNode(inputs);
    Adam optimizer(m.get_parameters(), lr);

    auto loss = m.forward(n_inputs)->CrossEntropy(labels);
    loss->backward();
    zero_grad();
    optimizer.clip_grad(1.0f);
    optimizer.step();

    allocMemAndInitTensors();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        gDoActions();
        std::cout << "loss : " << g_backend_ops->get_float(loss->get_tensor(), 0) << std::endl;
    }
}

int main(int argc, char *argv[]) {
    int opt;
    int epochs = 30;
    int batch_size = 32;
    float lr = 0.001;

    while ((opt = getopt(argc, argv, "e:l:b:")) != -1) {
        switch (opt) {
            case 'e':
                epochs = atoi(optarg);
                break;
            case 'l':
                lr = atof(optarg);
                break;
            case 'b':
                batch_size = atoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f <corpus> -c <checpoint> -e <epochs>" << std::endl;
                return 1;
        }
    }
    construct_env();
    if (epochs > 0) {
        train(epochs, lr, batch_size);
    } else {
        // serving
    }
    destruct_env();
    return 0;
}