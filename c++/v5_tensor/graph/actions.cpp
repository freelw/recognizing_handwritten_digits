#include "actions.h"
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <sstream>
#include "backends/backend_ops.h"

std::ostream &operator<<(std::ostream &output, const Action &a) {
    output << a.to_string();
    return output;
}

void AddAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->add(lhs, rhs, res);
}

std::string AddAction::to_string() const {
    std::ostringstream oss;
    oss << "AddAction: " << *lhs << " + " << *rhs << " -> " << *res;
    return oss.str();
}

void AddEqAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    g_backend_ops->addEq(lhs, rhs);
}

std::string AddEqAction::to_string() const {
    std::ostringstream oss;
    oss << "AddEqAction: " << *lhs << " += " << *rhs;
    return oss.str();
}

void ExpandAddAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->expandAdd(lhs, rhs, res);
}

std::string ExpandAddAction::to_string() const {
    std::ostringstream oss;
    oss << "ExpandAddAction: " << *lhs << " + " << *rhs << " -> " << *res;
    return oss.str();
}

void AtAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->at(lhs, rhs, res);
}

std::string AtAction::to_string() const {
    std::ostringstream oss;
    oss << "AtAction: " << *lhs << " at " << *rhs << " -> " << *res;
    return oss.str();
}

void MulAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->mul(lhs, rhs, res);
}

std::string MulAction::to_string() const {
    std::ostringstream oss;
    oss << "MulAction: " << *lhs << " * " << *rhs << " -> " << *res;
    return oss.str();
}

void SumAction::execute() {
    assert(lhs != nullptr);
    assert(rhs == nullptr);
    if (dim < 0 || dim >= lhs->get_rank()) {
        std::cerr << "Error: Invalid dimension for sum operation" << std::endl;
        abort();
    }
    g_backend_ops->sum(lhs, res, dim);
}

std::string SumAction::to_string() const {
    std::ostringstream oss;
    oss << "SumAction: " << *lhs << " -> " << *res << " along dim " << dim;
    return oss.str();
}

void ReluAction::execute() {
    assert(lhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->relu(lhs, res);
}

std::string ReluAction::to_string() const {
    std::ostringstream oss;
    oss << "ReluAction: " << *lhs << " -> " << *res;
    return oss.str();
}

void ReluPrimeAction::execute() {
    assert(lhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->reluPrime(lhs, res);
}

std::string ReluPrimeAction::to_string() const {
    std::ostringstream oss;
    oss << "ReluPrimeAction: " << *lhs << " -> " << *res;
    return oss.str();
}

std::vector<Action*> g_actions;

void gCreateAction(Action *action) {
    g_actions.push_back(action);
}

void printAllActions() {
    std::cout << "Actions:" << std::endl;
    for (Action *action : g_actions) {
        std::cout << *action << std::endl;
    }
}

void freeAllActions() {
    for (Action *action : g_actions) {
        delete action;
    }
    g_actions.clear();
}
