#include <iostream>
#include <unistd.h>
#include "common.h"
#include "tensor.h"
#include "backends/cpu/cpu_ops.h"
#include "graph/node.h"
#include "dataloader/mnist_loader_base.h"
#include "optimizers/adam.h"
#include "model/mlp.h"

#define INPUT_LAYER_SIZE 784
#define TRAIN_IMAGES_NUM 50000
#define TEST_IMAGES_NUM 10000

void print_progress(uint i, uint tot) {
    std::cout << "\r[" << i << "/" << tot << "]" << std::flush;
}

void assign_inputs(
    Tensor *inputs, float *tmp_buffer,
    int offset,
    int batch_size,
    const std::vector<std::vector<unsigned char>> &data) {
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < INPUT_LAYER_SIZE; ++j) {
            tmp_buffer[i * INPUT_LAYER_SIZE + j] = static_cast<float>(data[offset + i][j]) / 256.0f;
        }
    }
    g_backend_ops->cp_to_device(
        inputs,
        reinterpret_cast<char*>(tmp_buffer),
        inputs->size()
    );
}

void assign_labels(
    Tensor *labels, int32_t *tmp_buffer,
    int offset,
    int batch_size,
    const std::vector<unsigned char> & train_labels) {

    assert(labels->get_dtype() == INT32);
    for (int i = 0; i < batch_size; ++i) {
        tmp_buffer[i] = static_cast<int32_t>(train_labels[offset + i]);
    }
    g_backend_ops->cp_to_device(
        labels,
        reinterpret_cast<char*>(tmp_buffer),
        labels->size()
    );
}

void train(int epochs, float lr, int batch_size) {
    MnistLoaderBase loader;
    loader.load();
    std::cout << "data loaded." << std::endl;

    MLP m(INPUT_LAYER_SIZE, {30, 10});
    Tensor *inputs = allocTensor({batch_size, 784}, "inputs");
    Tensor *labels = allocTensor({batch_size}, "labels", INT32);
    auto n_inputs = graph::allocNode(inputs);
    Adam optimizer(m.get_parameters(), lr);

    auto loss = m.forward(n_inputs)->CrossEntropy(labels);
    zero_grad();
    insert_boundary_action();
    loss->backward();
    optimizer.clip_grad(1.0f);
    optimizer.step();
    printAllActions();

    allocMemAndInitTensors();
    float *inputs_tmp_buffer = static_cast<float*>(::malloc(inputs->size()));
    int32_t *labels_tmp_buffer = static_cast<int32_t*>(::malloc(labels->size()));
    const std::vector<std::vector<unsigned char>> & train_images = loader.getTrainImages();
    const std::vector<unsigned char> & train_labels = loader.getTrainLabels();
    assert(train_images.size() % batch_size == 0);
    assert(TRAIN_IMAGES_NUM + TEST_IMAGES_NUM == train_images.size());
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float loss_sum = 0;
        int offset = 0;
        int loop_times = 0;
        std::cout << "epoch : " << epoch << std::endl;
        print_progress(offset, TRAIN_IMAGES_NUM);
        while (offset < TRAIN_IMAGES_NUM) {
            assign_inputs(
                inputs,
                static_cast<float*>(inputs_tmp_buffer),
                offset,
                batch_size,
                train_images
            );
            assign_labels(
                labels,
                labels_tmp_buffer,
                offset,
                batch_size,
                train_labels
            );
            offset += batch_size;
            gDoActions();
            loss_sum += g_backend_ops->get_float(loss->get_tensor(), 0);
            loop_times++;
            print_progress(offset, TRAIN_IMAGES_NUM);
        }
        std::cout << " loss : " << loss_sum / loop_times << std::endl;

        // evaluate
        offset = TRAIN_IMAGES_NUM;
        std::cout << "evaluating : " << std::endl;
        print_progress(offset-TRAIN_IMAGES_NUM, TEST_IMAGES_NUM);
        while (offset < train_images.size()) {
            assign_inputs(
                inputs,
                static_cast<float*>(inputs_tmp_buffer),
                offset,
                batch_size,
                train_images
            );
            offset += batch_size;
            gDoForwardActions();
            print_progress(offset-TRAIN_IMAGES_NUM, TEST_IMAGES_NUM);
        }
        std::cout << std::endl;
    }
    ::free(labels_tmp_buffer);
    ::free(inputs_tmp_buffer);
}

int main(int argc, char *argv[]) {
    int opt;
    int epochs = 30;
    int batch_size = 100;
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