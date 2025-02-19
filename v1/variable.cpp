#include "variable.h"
#include <cmath>

Variable::Variable() : value(0), gradient(0) {}

Variable::Variable(double _value) : value(_value), gradient(0) {}

Variable::Variable(double _value, double _gradient) : value(_value), gradient(_gradient) {}

VariablePtr Variable::operator+(VariablePtr p) {
    return std::make_shared<Variable>(value + p->value, gradient + p->gradient);
}

VariablePtr Variable::operator*(VariablePtr p) {
    return std::make_shared<Variable>(value * p->value, gradient * p->value + value * p->gradient);
}

VariablePtr Variable::Relu() {
    if (value > 0) {
        return std::make_shared<Variable>(value, gradient);
    } else {
        return std::make_shared<Variable>(0, 0);
    }
}

VariablePtr Variable::log() {
    return std::make_shared<Variable>(std::log(value));
}

VariablePtr Variable::exp() {
    return std::make_shared<Variable>(std::exp(value));
}

std::ostream & operator<<(std::ostream &output, const Variable &s) {
    output << s.value;
    return output;
}
