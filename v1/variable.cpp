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

Variable::Variable() : value(0), gradient(0), inputCount(0), m(0), v(0) {}

Variable::Variable(double _value) : value(_value), gradient(0), inputCount(0), m(0), v(0)  {}

Variable::Variable(double _value, double _gradient) : value(_value), gradient(_gradient), inputCount(0), m(0), v(0)  {}

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


VariablePtr Variable::operator/(VariablePtr p) {
    auto ret = new DivRes(this, p);
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

void Variable::dfs(int depth) {
    // std::cout << this << " : " << value << " " << gradient << std::endl;
    std::cout << "--> " << this;
    if (parents.size() == 0) {
        std::cout << std::endl;
    }
    for (auto parent : parents) {
        parent->dfs(depth+1);
        std::cout << std::string(depth*2, ' ') << "--> " << this << std::endl;
    }
}

void Variable::zeroGrad() {
    gradient = 0;
}

void Variable::adamUpdate(double lr, double beta1, double beta2, double epsilon, int t) {
    // std::cout << "adamUpdate : " << this << " : " << gradient << std::endl;
    
    /*
    p.m = self.beta1 * p.m + (1 - self.beta1) * p.grad
    p.v = self.beta2 * p.v + (1 - self.beta2) * (p.grad ** 2)
    m_hat = p.m / (1 - self.beta1 ** self.t)
    v_hat = p.v / (1 - self.beta2 ** self.t)
    p.data -= self.lr * (m_hat / (v_hat ** 0.5 + 1e-8) + self.weight_decay * p.data)
    */

    m = beta1 * m + (1 - beta1) * gradient;
    v = beta2 * v + (1 - beta2) * gradient * gradient;
    double m_hat = m / (1 - std::pow(beta1, t));
    double v_hat = v / (1 - std::pow(beta2, t));
    // std::cout << "t before : " << t << std::endl;
    // std::cout << "beta1 before : " << beta1 << std::endl;
    // std::cout << "beta2 before : " << beta2 << std::endl;
    // std::cout << "value before : " << value << std::endl;
    // std::cout << "m_hat before : " << m_hat << std::endl;
    // std::cout << "v_hat before : " << v_hat << std::endl;
    // std::cout << "epsilon before : " << epsilon << std::endl;
    value -= lr * (m_hat / (std::sqrt(v_hat) + epsilon));
    //std::cout << "value after : " << value << std::endl;
}

TmpVar::TmpVar() : Variable() {}

TmpVar::TmpVar(double _value) : Variable(_value) {}

TmpVar::TmpVar(double _value, double _gradient) : Variable(_value, _gradient) {}

void TmpVar::backward() {
    for (auto parent : parents) {
        parent->incGradient(gradient);
        parent->decInputCount();
    }
}

Parameter::Parameter() : Variable() {}

Parameter::Parameter(double _value) : Variable(_value) {}

Parameter::Parameter(double _value, double _gradient) : Variable(_value, _gradient) {}

void Parameter::backward() {
    for (auto parent : parents) {
        parent->incGradient(gradient);
        parent->decInputCount();
    }
}

AddRes::AddRes(VariablePtr _x, VariablePtr _y) {
    this->parents.push_back(_x);
    this->parents.push_back(_y);
    this->value = _x->getValue() + _y->getValue();
    _x->incInputCount();
    _y->incInputCount();
}

void AddRes::backward() {
    // std::cout << this << " : AddRes::backward() grad : " << gradient << " inputCount : " << inputCount <<std::endl;
    for (auto parent : parents) {
        // std::cout << "parent : " << parent << std::endl;
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
    // std::cout << this << " : MulRes::backward() grad : " << gradient << std::endl;
    assert(parents.size() == 2);
    auto x = parents[0];
    auto y = parents[1];
    x->incGradient(gradient * y->getValue());
    y->incGradient(gradient * x->getValue());
    x->decInputCount();
    y->decInputCount();
}

DivRes::DivRes(VariablePtr _x, VariablePtr _y) {
    this->parents.push_back(_x);
    this->parents.push_back(_y);
    this->value = _x->getValue() / _y->getValue();
    _x->incInputCount();
    _y->incInputCount();
}

void DivRes::backward() {
    // std::cout << this << " : DivRes::backward() grad : " << gradient << std::endl;
    assert(parents.size() == 2);
    auto x = parents[0];
    auto y = parents[1];
    x->incGradient(gradient / y->getValue());
    y->incGradient(-gradient * x->getValue() / (y->getValue() * y->getValue()));
    x->decInputCount();
    y->decInputCount();
}

ReluRes::ReluRes(VariablePtr _x) {
    this->parents.push_back(_x);
    this->value = _x->getValue() > 0 ? _x->getValue() : 0;
    _x->incInputCount();
}

void ReluRes::backward() {
    // std::cout << "ReluRes::backward() grad : " << gradient << std::endl;
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
    // std::cout << "LogRes::backward() grad : " << gradient << std::endl;
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
    // std::cout << "ExpRes::backward() grad : " << gradient << " value : " << value << std::endl;
    assert(parents.size() == 1);
    auto x = parents[0];
    // std::cout << x << std::endl;
    x->incGradient(gradient * value);
    x->decInputCount();
}
