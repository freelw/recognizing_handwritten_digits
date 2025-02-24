
#include "mlp.h"
#include <iostream>

void testgrad() {

    Model m(4, {5, 3}, false);
    m.zeroGrad();

    std::vector<VariablePtr> input;
    for (int i = 0; i < 4; i++) {
        input.emplace_back(allocTmpVar(i));
    }

    std::vector<VariablePtr> res = m.forward(input);

    std::cout << "forward result: ";
    for (uint i = 0; i < res.size(); i++) {
        std::cout << res[i]->getValue() << " ";
    }
    std::cout << std::endl;
    VariablePtr loss = CrossEntropyLoss(res, 1);

    std::cout << "loss: " << loss->getValue() << std::endl;

    loss->setGradient(1);
    loss->bp();
    // m.update(0.01, 1);

    std::cout << m << std::endl;
}
