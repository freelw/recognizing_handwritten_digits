#include "variable.h"
#include <cassert>
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

Variable::Variable() : value(0), gradient(0), inputCount(0) {}

Variable::Variable(double _value) : value(_value), gradient(0), inputCount(0) {}

Variable::Variable(double _value, double _gradient) : value(_value), gradient(_gradient), inputCount(0) {}

VariablePtr Variable::operator+(VariablePtr p) {
    auto ret = new AddRes(this, p);
    registerTmpVar(ret);
    return ret;
}

VariablePtr Variable::operator*(VariablePtr p) {
    auto ret = new MulRes(this, p);
    registerTmpVar(ret);
    return ret;
}

VariablePtr Variable::Relu() {
    auto ret = new ReluRes(this);
    registerTmpVar(ret);
    return ret;
}

VariablePtr Variable::log() {
    auto ret = new LogRes(this);
    registerTmpVar(ret);
    return ret;
}

VariablePtr Variable::exp() {
    auto ret = new ExpRes(this);
    registerTmpVar(ret);
    return ret;
}

std::ostream & operator<<(std::ostream &output, const Variable &s) {
    output << s.value << " " << s.gradient;
    return output;
}

void Variable::bp() {
    this->backward();
    for (auto parent : parents) {
        if (parent->inputCount == 0) {
            parent->bp();
        }
    }
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
    _x->incInputCount();
    _y->incInputCount();
}

void AddRes::backward() {
    for (auto parent : parents) {
        parent->incGradient(gradient);
        parent->decInputCount();
    }
}

MulRes::MulRes(VariablePtr _x, VariablePtr _y) {
    this->parents.push_back(_x);
    this->parents.push_back(_y);
    this->value = _x->getValue() * _y->getValue();
    _x->incInputCount();
    _y->incInputCount();
}

void MulRes::backward() {
    assert(parents.size() == 2);
    auto x = parents[0];
    auto y = parents[1];
    x->incGradient(gradient * y->getValue());
    y->incGradient(gradient * x->getValue());
    x->decInputCount();
    y->decInputCount();
}

ReluRes::ReluRes(VariablePtr _x) {
    this->parents.push_back(_x);
    this->value = _x->getValue() > 0 ? _x->getValue() : 0;
    _x->incInputCount();
}

void ReluRes::backward() {
    assert(parents.size() == 1);
    auto x = parents[0];
    x->incGradient(gradient * (x->getValue() > 0 ? 1 : 0));
    x->decInputCount();
}

LogRes::LogRes(VariablePtr _x) {
    this->parents.push_back(_x);
    this->value = std::log(_x->getValue());
    _x->incInputCount();
}

void LogRes::backward() {
    assert(parents.size() == 1);
    auto x = parents[0];
    x->incGradient(gradient / x->getValue());
    x->decInputCount();
}

ExpRes::ExpRes(VariablePtr _x) {
    this->parents.push_back(_x);
    this->value = std::exp(_x->getValue());
    _x->incInputCount();
}

void ExpRes::backward() {
    assert(parents.size() == 1);
    auto x = parents[0];
    x->incGradient(gradient * value);
    x->decInputCount();
}
