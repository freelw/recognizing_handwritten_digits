#include "actions.h"
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
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
    return "AddAction: " + lhs->get_name() + " + " + rhs->get_name() + " -> " + res->get_name();
}

void AddEqAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    g_backend_ops->addEq(lhs, rhs);
}

std::string AddEqAction::to_string() const {
    return "AddEqAction: " + lhs->get_name() + " += " + rhs->get_name();
}

void ExpandAddAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->expandAdd(lhs, rhs, res);
}

std::string ExpandAddAction::to_string() const {
    return "ExpandAddAction: " + lhs->get_name() + " + " + rhs->get_name() + " -> " + res->get_name();
}

void AtAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->at(lhs, rhs, res);
}

std::string AtAction::to_string() const {
    return "AtAction: " + lhs->get_name() + " at " + rhs->get_name() + " -> " + res->get_name();
}

void MulAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->mul(lhs, rhs, res);
}

std::string MulAction::to_string() const {
    return "MulAction: " + lhs->get_name() + " * " + rhs->get_name() + " -> " + res->get_name();
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
    return "SumAction: " + lhs->get_name() + " -> " + res->get_name() + " along dim " + std::to_string(dim);
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