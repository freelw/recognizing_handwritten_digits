#include "variable.h"

#include <iostream>

int main() {
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
    return 0;
}