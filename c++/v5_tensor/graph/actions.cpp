#include "actions.h"
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <sstream>
#include "backends/backend_ops.h"
#include "optimizers/parameter.h"

extern bool g_training;

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
    oss << "AddAction: " << lhs->get_meta_info() << " + " << rhs->get_meta_info() << " -> " << res->get_meta_info();
    return oss.str();
}

AddEqAction::AddEqAction(Tensor *_lhs, const Tensor *_rhs)
    : Action(_lhs, _rhs, nullptr) {
    assert(_lhs->get_dim() == _rhs->get_dim());
    assert(_lhs->get_shape() == _rhs->get_shape());
    auto dim = _lhs->get_dim();
    lhs_shape = allocTensor(
        {dim},
        _lhs->get_name() + "_shape",
        INT32
    );
    lhs_strides = allocTensor(
        {dim},
        _lhs->get_name() + "_strides",
        INT32
    );
    rhs_strides = allocTensor(
        {dim},
        _rhs->get_name() + "_strides",
        INT32
    );
    gCreateAction(
        new AssignShapeAndStridesAction(
            lhs_shape,
            lhs_strides,
            _lhs->get_shape(),
            _lhs->get_strides()
        )
    );
    gCreateAction(
        new AssignShapeAndStridesAction(
            nullptr,
            rhs_strides,
            {},
            _rhs->get_strides()
        )
    );
}

void AddEqAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(lhs->get_shape() == rhs->get_shape());
    g_backend_ops->addEq(lhs, rhs, lhs_shape, lhs_strides, rhs_strides);    
}

std::string AddEqAction::to_string() const {
    std::ostringstream oss;
    oss << "AddEqAction: " << lhs->get_meta_info() << " += " << rhs->get_meta_info();
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
    oss << "ExpandAddAction: " << lhs->get_meta_info() << " + " << rhs->get_meta_info() << " -> " << res->get_meta_info();
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
    oss << "AtAction: " << lhs->get_meta_info() << " at " << rhs->get_meta_info() << " -> " << res->get_meta_info();
    return oss.str();
}

MulAction::MulAction(Tensor *_lhs, const Tensor *_rhs, Tensor *_res)
    : Action(_lhs, _rhs, _res) {
    assert(_lhs->get_dim() == _rhs->get_dim());
    assert(_lhs->get_shape() == _rhs->get_shape());
    auto dim = _lhs->get_dim();

    lhs_shape = allocTensor(
        {dim},
        _lhs->get_name() + "_shape",
        INT32
    );
    lhs_strides = allocTensor(
        {dim},
        _lhs->get_name() + "_strides",
        INT32
    );
    rhs_strides = allocTensor(
        {dim},
        _rhs->get_name() + "_strides",
        INT32
    );
    res_strides = allocTensor(
        {dim},
        _res->get_name() + "_strides",
        INT32
    );

    gCreateAction(
        new AssignShapeAndStridesAction(
            lhs_shape,
            lhs_strides,
            _lhs->get_shape(),
            _lhs->get_strides()
        )
    );

    gCreateAction(
        new AssignShapeAndStridesAction(
            nullptr,
            rhs_strides,
            {},
            _rhs->get_strides()
        )
    );

    gCreateAction(
        new AssignShapeAndStridesAction(
            nullptr,
            res_strides,
            {},
            _res->get_strides()
        )
    );
}

void MulAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->mul(
        lhs, rhs, res,
        lhs_shape, lhs_strides,
        rhs_strides, res_strides
    );
}

std::string MulAction::to_string() const {
    std::ostringstream oss;
    oss << "MulAction: " << lhs->get_meta_info() << " * " << rhs->get_meta_info() << " -> " << res->get_meta_info();
    return oss.str();
}

void SumAction::execute() {
    assert(lhs != nullptr);
    assert(rhs == nullptr);
    if (dim < 0 || dim >= lhs->get_dim()) {
        std::cerr << "Error: Invalid dimension for sum operation" << std::endl;
        abort();
    }
    g_backend_ops->sum(lhs, res, dim);
}

std::string SumAction::to_string() const {
    std::ostringstream oss;
    oss << "SumAction: " << lhs->get_meta_info() << " -> " << res->get_meta_info() << " along dim " << dim;
    return oss.str();
}

void ReluAction::execute() {
    assert(lhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->relu(lhs, res);
}

std::string ReluAction::to_string() const {
    std::ostringstream oss;
    oss << "ReluAction: " << lhs->get_meta_info() << " -> " << res->get_meta_info();
    return oss.str();
}

void ReluPrimeAction::execute() {
    assert(lhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->reluPrime(lhs, res);
}

std::string ReluPrimeAction::to_string() const {
    std::ostringstream oss;
    oss << "ReluPrimeAction: " << lhs->get_meta_info() << " -> " << res->get_meta_info();
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
    oss << "CrossEntropyAction: " << 
        lhs->get_meta_info() << " with labels " <<
        rhs->get_meta_info() << " -> " <<
        res->get_meta_info() << " context : " <<
        maxs->get_meta_info() << ", " <<
        sums->get_meta_info();
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
    oss << "CrossEntropyBackwardAction: " << 
        lhs->get_meta_info() <<" with labels " <<
        rhs->get_meta_info() << " -> " <<
        res->get_meta_info() <<" context : " <<
        maxs->get_meta_info() << ", " <<
        sums->get_meta_info();
    return oss.str();
}

void CalcAllGradNormAction::execute() {
    assert(res != nullptr);
    g_backend_ops->calcAllGradNorm(grads, res);
}

std::string CalcAllGradNormAction::to_string() const {
    std::ostringstream oss;
    oss << "CalcAllGradNormAction: calculating gradient norm for " << grads.size() << " tensors" << " -> " << res->get_meta_info();
    return oss.str();
}

void ClipGradAction::execute() {
    assert(lhs != nullptr); // grad
    assert(rhs != nullptr); // norm
    g_backend_ops->clipGrad(lhs, rhs, grad_clip_val);
}

std::string ClipGradAction::to_string() const {
    std::ostringstream oss;
    oss << "ClipGradAction: clipping gradient " << lhs->get_meta_info() << " with norm " << rhs->get_meta_info() << " to grad_clip_val: " << grad_clip_val;
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
    oss << "AdamStepAction: updating parameter " << param->get_w()->get_meta_info() << " with learning rate " << lr;
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
        g_backend_ops->init_weight_for_dbg(lhs, sigma);
    } else if (init_type == "fill") {
        g_backend_ops->fill(lhs, sigma);
    } else {
        std::cerr << "Error: Unknown initialization type: " << init_type << std::endl;
        abort();
    }
}

std::string InitWeightAction::to_string() const {
    std::ostringstream oss;
    oss << "InitWeightAction: initializing " << lhs->get_meta_info() 
        << " with type " << init_type
        << " sigma " << sigma
        << " mean " << mean;
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

AssignShapeAndStridesAction::AssignShapeAndStridesAction(
    Tensor *tensor_shape,
    Tensor *tensor_strides,
    const std::vector<int> &shape,
    const std::vector<int> &strides
) : Action(tensor_shape, nullptr, tensor_strides) {
    if (tensor_shape != nullptr) {    
        shape_data = static_cast<int32_t*>(::malloc(sizeof(int32_t) * shape.size()));
        for (size_t i = 0; i < shape.size(); ++i) {
            shape_data[i] = static_cast<int32_t>(shape[i]);
        }
    }
    if (tensor_strides != nullptr) {
        strides_data = static_cast<int32_t*>(::malloc(sizeof(int32_t) * strides.size()));
        for (size_t i = 0; i < strides.size(); ++i) {
            strides_data[i] = static_cast<int32_t>(strides[i]);
        }
    }
}

AssignShapeAndStridesAction::~AssignShapeAndStridesAction() {
    if (lhs != nullptr) {
        assert(shape_data != nullptr);
        ::free(shape_data);
    }
    if (res != nullptr) {
        assert(strides_data != nullptr);
        ::free(strides_data);
    }
}

void AssignShapeAndStridesAction::execute() {
    if (lhs != nullptr) {
        assert(shape_data != nullptr);
        g_backend_ops->cp_to_device(
            lhs,
            reinterpret_cast<char*>(shape_data),
            lhs->size()
        );
    }
    if (res != nullptr) {
        assert(strides_data != nullptr);
        g_backend_ops->cp_to_device(
            res,
            reinterpret_cast<char*>(strides_data),
            res->size()
        );
    }
}

std::string AssignShapeAndStridesAction::to_string() const {
    std::ostringstream oss;
    oss << "AssignShapeAndStridesAction: assigning shape " << (lhs == nullptr ? "null" : lhs->get_meta_info())
        << " and strides " << (res == nullptr ? "null" : res->get_meta_info());
    return oss.str();
}

void ReshapeDeepCpAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    g_backend_ops->reshape_deep_cp(lhs, rhs, shape, strides);
}

std::string ReshapeDeepCpAction::to_string() const {
    std::ostringstream oss;
    oss << "ReshapeDeepCpAction: deep copy " << lhs->get_meta_info() << " to " << rhs->get_meta_info() << " with strides " << strides->get_meta_info();
    return oss.str();
}

void RepeatInterleaveAction::execute() {
    assert(lhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->repeat_interleave(lhs, res, n);
}

std::string RepeatInterleaveAction::to_string() const {
    std::ostringstream oss;
    oss << "RepeatInterleaveAction: repeat interleave " << lhs->get_meta_info() << " to " << res->get_meta_info() << " with n = " << n;
    return oss.str();
}

void SequenceMaskAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->sequence_mask(lhs, rhs, res, value);
}

std::string SequenceMaskAction::to_string() const {
    std::ostringstream oss;
    oss << "SequenceMaskAction: sequence mask " << lhs->get_meta_info() << " with mask " << rhs->get_meta_info() << " to " << res->get_meta_info();
    return oss.str();
}

void SoftmaxAction::execute() {
    assert(lhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->softmax(lhs, res);
}

std::string SoftmaxAction::to_string() const {
    std::ostringstream oss;
    oss << "SoftmaxAction: softmax " << lhs->get_meta_info() << " to " << res->get_meta_info();
    return oss.str();
}

void SoftmaxBackwardAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    // lhs is target grad
    // rhs is softmax result
    // res is input grad
    g_backend_ops->softmax_bacward(lhs, rhs, res);
}

std::string SoftmaxBackwardAction::to_string() const {
    std::ostringstream oss;
    oss << "SoftmaxBackwardAction: softmax backward " << res->get_meta_info() << " with softmax result " << rhs->get_meta_info() << " to " << lhs->get_meta_info();
    return oss.str();
}

void DivAction::execute() {
    assert(lhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->div(res, lhs, value);
}

std::string DivAction::to_string() const {
    std::ostringstream oss;
    oss << "DivAction: dividing " << lhs->get_meta_info() << " by " << value << " to " << res->get_meta_info();
    return oss.str();
}

DropoutMaskAction::DropoutMaskAction(Tensor *mask, float _p)
    : Action(nullptr, nullptr, mask), p(_p) {
    assert(mask != nullptr);
    shape = allocTensor(
        {mask->get_dim()},
        mask->get_name() + "_shape",
        INT32
    );
    strides = allocTensor(
        {mask->get_dim()},
        mask->get_name() + "_strides",
        INT32
    );
    gCreateAction(
        new AssignShapeAndStridesAction(
            shape,
            strides,
            mask->get_shape(),
            mask->get_strides()
        )
    );
}

void DropoutMaskAction::execute() {
    assert(res != nullptr);
    g_backend_ops->build_dropout_mask(res, p, shape, strides);
}

std::string DropoutMaskAction::to_string() const {
    std::ostringstream oss;
    oss << "Build Dropout mask: " << res->get_meta_info() << " with rate " << p;
    return oss.str();
}

void EmbeddingAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->embedding(lhs, rhs, res);
}

std::string EmbeddingAction::to_string() const {
    std::ostringstream oss;
    oss << "EmbeddingAction: embedding " << lhs->get_meta_info() << " with indices " << rhs->get_meta_info() << " to " << res->get_meta_info();
    return oss.str();
}

void EmbeddingBackwardAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->embeddingBackward(lhs, rhs, res);
}

std::string EmbeddingBackwardAction::to_string() const {
    std::ostringstream oss;
    oss << "EmbeddingBackwardAction: embedding backward " << lhs->get_meta_info() << " with indices " << rhs->get_meta_info() << " to " << res->get_meta_info();
    return oss.str();
}

void PosEncodingAction::execute() {
    assert(res != nullptr);
    g_backend_ops->pos_encoding(res);
}

std::string PosEncodingAction::to_string() const {
    std::ostringstream oss;
    auto shape = res->get_shape();
    oss << "PosEncodingAction: position encoding " << res->get_meta_info() << " with max_len " << shape[0] << " and num_hidden " << shape[1];
    return oss.str();
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

bool validateBoundaryFound() {
    bool boundary_found = false;
    for (Action *action : g_actions) {
        if (action->is_backward_boundary()) {
            boundary_found = true;
        }
    }
    return boundary_found;
}

bool validateAddEqActionsInBackward() {
    // todo: all AddEqAction should be in backward(behind boundary)
    return true;
}

void gDoActions() {
    assert(validateBoundaryFound());
    assert(validateAddEqActionsInBackward());
    g_training = true;
    for (Action *action : g_actions) {
        if (action->is_do_once() && action->executed_once()) {
            continue;
        }
        action->execute();
        action->increase_exec_times();
    }
}

void gDoForwardActions() {
    g_training = false;
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

void disableOnceAction() {
    for (Action *action : g_actions) {
        if (action->is_do_once()) {
            action->increase_exec_times();
        }
    }
}
