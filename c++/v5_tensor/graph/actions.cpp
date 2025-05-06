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
    assert(lhs->get_shape() == rhs->get_shape());
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

void CrossEntropyAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    assert(maxs != nullptr);
    assert(sums != nullptr);
    g_backend_ops->crossEntropy(lhs, rhs, maxs, sums, res);
}

std::string CrossEntropyAction::to_string() const {
    std::ostringstream oss;
    oss << "CrossEntropyAction: " << *lhs << " with labels " << *rhs << " -> " << *res << " context : " << *maxs << ", " << *sums;
    return oss.str();
}

void CrossEntropyBackwardAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    assert(maxs != nullptr);
    assert(sums != nullptr);
    // lhs is zt
    // rhs is labels
    // res is grad
    g_backend_ops->crossEntropyBackward(lhs, rhs, maxs, sums, res);
}

std::string CrossEntropyBackwardAction::to_string() const {
    std::ostringstream oss;
    oss << "CrossEntropyBackwardAction: " << *lhs << " with labels " << *rhs << " -> " << *res << " context : " << *maxs << ", " << *sums;
    return oss.str();
}

void ZeroGradAction::execute() {
    g_backend_ops->memset(grad_tensors_data, 0, grad_tensors_data_capacity);
}

std::string ZeroGradAction::to_string() const {
    return "ZeroGradAction: zeroing gradients";
}

std::vector<Action*> g_actions;

void gCreateAction(Action *action) {
    g_actions.push_back(action);
}

void gDoActions() {
    for (Action *action : g_actions) {
        action->execute();
    }
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
