#include "actions.h"
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <sstream>
#include "backends/backend_ops.h"
#include "optimizers/parameter.h"

bool Action::is_do_once() const {
    return false;
}

bool Action::is_backward_boundary() {
    return false;
}

bool Action::executed_once() const {
    return exec_times > 0;
}

void Action::increase_exec_times() {
    exec_times++;
}

int Action::get_exec_times() const {
    return exec_times;
}

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

void CalcAllGradNormAction::execute() {
    assert(res != nullptr);
    g_backend_ops->calcAllGradNorm(grads, res);
}

std::string CalcAllGradNormAction::to_string() const {
    std::ostringstream oss;
    oss << "CalcAllGradNormAction: calculating gradient norm for " << grads.size() << " tensors" << " -> " << *res;
    return oss.str();
}

void ClipGradAction::execute() {
    assert(lhs != nullptr); // grad
    assert(rhs != nullptr); // norm
    g_backend_ops->clipGrad(lhs, rhs, grad_clip_val);
}

std::string ClipGradAction::to_string() const {
    std::ostringstream oss;
    oss << "ClipGradAction: clipping gradient " << *lhs << " with norm " << *rhs << " to grad_clip_val: " << grad_clip_val;
    return oss.str();
}

void AdamStepAction::execute() {
    param->inc_t();
    int t = param->get_t();
    Tensor *w = param->get_w();
    Tensor *grad = param->get_grad();
    Tensor *m = param->get_m();
    Tensor *v = param->get_v();

    g_backend_ops->adamStep(w, grad, m, v, t, lr, beta1, beta2, epsilon);
}

std::string AdamStepAction::to_string() const {
    std::ostringstream oss;
    oss << "AdamStepAction: updating parameter " << *param->get_w() << " with learning rate " << lr;
    return oss.str();
}

void ZeroGradAction::execute() {
    g_backend_ops->memset(grad_tensors_data, 0, grad_tensors_data_capacity);
}

std::string ZeroGradAction::to_string() const {
    return "ZeroGradAction: zeroing gradients";
}

void InitWeightAction::execute() {
    assert(lhs != nullptr);

    if (init_type == "gauss") {
        g_backend_ops->init_weight_gauss(lhs, mean, sigma);
    } else if (init_type == "uniform") {
        g_backend_ops->init_weight_uniform(lhs, sigma);
    } else if (init_type == "xavier") {
        assert(false);
        // g_backend_ops->xavier(lhs);
    } else if (init_type == "kaiming") {
        assert(false);
        // g_backend_ops->kaiming(lhs);
    } else if (init_type == "dbg") {
        
    } else {
        std::cerr << "Error: Unknown initialization type: " << init_type << std::endl;
        abort();
    }
}

std::string InitWeightAction::to_string() const {
    std::ostringstream oss;
    oss << "InitWeightAction: initializing " << *lhs << " with type " << init_type;
    return oss.str();
}

void BoundaryAction::execute() {
    // Do nothing
}

std::string BoundaryAction::to_string() const {
    return "============= BoundaryAction: boundary action =============";
}

bool BoundaryAction::is_backward_boundary() {
    return true;
}

std::vector<Action*> g_actions;

std::vector<Action *> getOnceActions() {
    std::vector<Action *> once_actions;
    for (Action *action : g_actions) {
        if (action->is_do_once() && !action->executed_once()) {
            once_actions.push_back(action);
        }
    }
    return once_actions;
}

void gCreateAction(Action *action) {
    g_actions.push_back(action);
}

void gDoActions() {
    for (Action *action : g_actions) {
        if (action->is_do_once() && action->executed_once()) {
            continue;
        }
        action->execute();
        action->increase_exec_times();
    }
}

void gDoForwardActions() {
    for (Action *action : g_actions) {
        if (action->is_do_once() && action->executed_once()) {
            continue;
        }
        if (action->is_backward_boundary()) {
            break;
        }
        action->execute();
        action->increase_exec_times();
    }
}

void printAllActions() {
    std::cout << "Actions:" << std::endl;
    for (Action *action : g_actions) {
        if (action->is_do_once()) {
            std::cout << "[once]";
        }
        std::cout << *action << std::endl;
    }
}

void freeAllActions() {
    for (Action *action : g_actions) {
        delete action;
    }
    g_actions.clear();
}
