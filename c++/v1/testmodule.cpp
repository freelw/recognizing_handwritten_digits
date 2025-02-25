
#include "mlp.h"
#include <iostream>

void testmodule() {

    std::vector<VariablePtr> input;

    for (int i = 0; i < 784; i++) {
        input.emplace_back(allocTmpVar(3));
    }

    std::vector<uint> sizes;
    sizes.push_back(30);
    sizes.push_back(10);

    Model m(784, sizes, false);

    auto res = m.forward(input);

    std::cout << "Model forward: ";
    std::cout.precision(10);
    for (auto p : res) {
        std::cout << p->getValue() << " ";
    }
    std::cout << std::endl;
}
