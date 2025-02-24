
#include "mlp.h"
#include <iostream>

void testcrossentropy() {

    std::vector<VariablePtr> input;

    for (int i = 0; i < 4; i++) {
        input.emplace_back(allocTmpVar(i));
    }

    auto res = CrossEntropyLoss(input, 1);

    std::cout << "CrossEntropyLoss: " << res->getValue() << std::endl;
}
