#include "variable.h"

#include <iostream>

int main() {
    VariablePtr x = std::make_shared<Variable>(1.0);
    VariablePtr y = std::make_shared<Variable>(2.0);
    VariablePtr z = *((*x) * y) + x;
    std::cout << *z << std::endl;
    z = z->Relu();
    std::cout << *z << std::endl;
    z = z->log();
    std::cout << *z << std::endl;
    z = z->exp();
    std::cout << *z << std::endl;
    return 0;
}