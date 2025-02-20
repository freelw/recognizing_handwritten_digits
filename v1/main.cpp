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

int main() {
    MnistLoaderBase loader;
    loader.load();

    std::cout << "data loaded." << std::endl;
    std::vector<int> sizes;
    sizes.push_back(30);
    sizes.push_back(10);
    Model m(INPUT_LAYER_SIZE, sizes);
    destroyTmpVars();
    return 0;
}