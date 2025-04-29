#include "actions.h"
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include "backends/backend_ops.h"

void AddAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    g_backend_ops->add(lhs, rhs, res);
}

void AddEqAction::execute() {
    g_backend_ops->addEq(lhs, rhs);
}

void AtAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    g_backend_ops->at(lhs, rhs, res);
}

void MulAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    g_backend_ops->mul(lhs, rhs, res);
}

void SumAction::execute() {
    assert(lhs != nullptr);
    assert(rhs == nullptr);
    if (dim < 0 || dim >= lhs->get_rank()) {
        std::cerr << "Error: Invalid dimension for sum operation" << std::endl;
        abort();
    }
    g_backend_ops->sum(lhs, dim, res);
}

std::vector<Action*> g_actions;

void gCreateAction(Action *action) {
    g_actions.push_back(action);
}

void freeAllActions() {
    for (Action *action : g_actions) {
        delete action;
    }
    g_actions.clear();
}