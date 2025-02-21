
#include "mlp.h"
#include <iostream>

void testgrad() {

    Model m(10, {20, 11}, false);
    m.zeroGrad();

    std::vector<VariablePtr> input;
    for (int i = 0; i < 10; i++) {
        input.emplace_back(allocTmpVar(i));
    }

    std::vector<VariablePtr> res = m.forward(input);
    VariablePtr loss = CrossEntropyLoss(res, 5);

    loss->setGradient(1);
    loss->bp();
    m.update(0.01, 1);

    std::cout << m << std::endl;
}
