#include "variable.h"
#include <cmath>


std::vector<VariablePtr> tmpVars;

void destroyTmpVars() {
    for (auto var : tmpVars) {
        delete var;
    }
    tmpVars.clear();
}

void registerTmpVar(VariablePtr var) {
    tmpVars.push_back(var);
}

Variable::Variable() : value(0), gradient(0) {}

Variable::Variable(double _value) : value(_value), gradient(0) {}

Variable::Variable(double _value, double _gradient) : value(_value), gradient(_gradient) {}

VariablePtr Variable::operator+(VariablePtr p) {
    auto ret = new AddRes(this, p);
    registerTmpVar(ret);
}

VariablePtr Variable::operator*(VariablePtr p) {
    // return std::make_shared<Variable>(value * p->value, gradient * p->value + value * p->gradient);
}

VariablePtr Variable::Relu() {
    if (value > 0) {
        // return std::make_shared<Variable>(value, gradient);
    } else {
        // return std::make_shared<Variable>(0, 0);
    }
}

VariablePtr Variable::log() {
    // return std::make_shared<Variable>(std::log(value));
}

VariablePtr Variable::exp() {
    // return std::make_shared<Variable>(std::exp(value));
}

std::ostream & operator<<(std::ostream &output, const Variable &s) {
    output << s.value;
    return output;
}

Parameter::Parameter() : Variable() {}

Parameter::Parameter(double _value) : Variable(_value) {}

Parameter::Parameter(double _value, double _gradient) : Variable(_value, _gradient) {}

void Parameter::backward() {
}

AddRes::AddRes(VariablePtr _x, VariablePtr _y) {
    this->parents.push_back(_x);
    this->parents.push_back(_y);
    this->value = _x->getValue() + _y->getValue();
}

void AddRes::backward() {
    for (auto parent : parents) {
        parent->incGradient(gradient);
    }
}
