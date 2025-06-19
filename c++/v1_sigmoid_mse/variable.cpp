#include "variable.h"
#include <cassert>
#include <cmath>
#include <iostream>

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

VariablePtr allocTmpVar(double value) {
    auto ret = new TmpVar(value);
    registerTmpVar(ret);
    return ret;
}

Variable::Variable() : value(0), gradient(0), inputCount(0) {
    parents[0] = nullptr;
    parents[1] = nullptr;
}

Variable::Variable(double _value) : value(_value), gradient(0), inputCount(0) {
    parents[0] = nullptr;
    parents[1] = nullptr;
}

Variable::Variable(double _value, double _gradient) : value(_value), gradient(_gradient), inputCount(0) {
    parents[0] = nullptr;
    parents[1] = nullptr;
}

VariablePtr Variable::operator+(VariablePtr p) {
    auto ret = new AddRes(this, p);
    registerTmpVar(ret);
    return ret;
}

VariablePtr Variable::operator-(VariablePtr p) {
    auto ret = new MinusRes(this, p);
    registerTmpVar(ret);
    return ret;
}

VariablePtr Variable::operator*(VariablePtr p) {
    auto ret = new MulRes(this, p);
    registerTmpVar(ret);
    return ret;
}

VariablePtr Variable::operator/(VariablePtr p) {
    auto ret = new DivRes(this, p);
    registerTmpVar(ret);
    return ret;
}

VariablePtr Variable::exp() {
    auto ret = new ExpRes(this);
    registerTmpVar(ret);
    return ret;
}

VariablePtr Variable::sqr() {
    auto ret = new SqrRes(this);
    registerTmpVar(ret);
    return ret;
}

VariablePtr Variable::sigmoid() {
    auto ret = new SigmoidRes(this);
    registerTmpVar(ret);
    return ret;
}

std::ostream& operator<<(std::ostream& output, const Variable& s) {
    output << s.value << " " << s.gradient;
    return output;
}

void Variable::bp() {
    this->backward();
    for (auto i = 0; i < 2; ++i) {
        auto parent = parents[i];
        if (parent) {
            if (parent->inputCount == 0) {
                parent->bp();
            }
        }
    }
}

void Variable::zeroGrad() {
    gradient = 0;
}

void Variable::update(double lr) {
    value -= lr * gradient;
}

TmpVar::TmpVar() : Variable() {}

TmpVar::TmpVar(double _value) : Variable(_value) {}

TmpVar::TmpVar(double _value, double _gradient) : Variable(_value, _gradient) {}

void TmpVar::backward() {
    for (auto i = 0; i < 2; ++i) {
        auto parent = parents[i];
        if (parent) {
            parent->incGradient(gradient);
            parent->decInputCount();
        }
    }
}

void Variable::backward() {
    for (auto i = 0; i < 2; ++i) {
        auto parent = parents[i];
        if (parent) {
            parent->incGradient(gradient);
            parent->decInputCount();
        }
    }
}

AddRes::AddRes(VariablePtr _x, VariablePtr _y) {
    parents[0] = _x;
    parents[1] = _y;
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

MinusRes::MinusRes(VariablePtr _x, VariablePtr _y) {
    parents[0] = _x;
    parents[1] = _y;
    this->value = _x->getValue() - _y->getValue();
    _x->incInputCount();
    _y->incInputCount();
}

void MinusRes::backward() {
    assert(parents[0] != nullptr && parents[1] != nullptr);
    auto x = parents[0];
    auto y = parents[1];
    x->incGradient(gradient);
    y->incGradient(-gradient);
    x->decInputCount();
    y->decInputCount();
}

MulRes::MulRes(VariablePtr _x, VariablePtr _y) {
    parents[0] = _x;
    parents[1] = _y;
    this->value = _x->getValue() * _y->getValue();
    _x->incInputCount();
    _y->incInputCount();
}

void MulRes::backward() {
    assert(parents[0] != nullptr && parents[1] != nullptr);
    auto x = parents[0];
    auto y = parents[1];
    x->incGradient(gradient * y->getValue());
    y->incGradient(gradient * x->getValue());
    x->decInputCount();
    y->decInputCount();
}

DivRes::DivRes(VariablePtr _x, VariablePtr _y) {
    parents[0] = _x;
    parents[1] = _y;
    this->value = _x->getValue() / _y->getValue();
    _x->incInputCount();
    _y->incInputCount();
}

void DivRes::backward() {
    assert(parents[0] != nullptr && parents[1] != nullptr);
    auto x = parents[0];
    auto y = parents[1];
    x->incGradient(gradient / y->getValue());
    y->incGradient(-gradient * x->getValue() / (y->getValue() * y->getValue()));
    x->decInputCount();
    y->decInputCount();
}

ExpRes::ExpRes(VariablePtr _x) {
    parents[0] = _x;
    this->value = std::exp(_x->getValue());
    _x->incInputCount();
}

void ExpRes::backward() {
    assert(parents[0] != nullptr && parents[1] == nullptr);
    auto x = parents[0];
    x->incGradient(gradient * value);
    x->decInputCount();
}

SqrRes::SqrRes(VariablePtr _x) {
    parents[0] = _x;
    this->value = _x->getValue() * _x->getValue();
    _x->incInputCount();
}

void SqrRes::backward() {
    assert(parents[0] != nullptr && parents[1] == nullptr);
    auto x = parents[0];
    x->incGradient(gradient * 2 * x->getValue());
    x->decInputCount();
}

SigmoidRes::SigmoidRes(VariablePtr _x) {
    parents[0] = _x;
    this->value = 1.0 / (1.0 + std::exp(-_x->getValue()));
    _x->incInputCount();
}

void SigmoidRes::backward() {
    assert(parents[0] != nullptr && parents[1] == nullptr);
    auto x = parents[0];
    double sigmoid_value = value;
    x->incGradient(gradient * sigmoid_value * (1 - sigmoid_value));
    x->decInputCount();
}