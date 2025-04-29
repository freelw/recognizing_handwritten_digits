#include "autograd/optimizers.cuh"
#include <iostream>
#include "backends/ops.cuh"

namespace autograd_cuda {
    void Adam::step() {

        cp_from_device();
        
        for (auto p : parameters) {
            if (!p->require_grad()) {
                continue;
            }
            p->inc_t();
            auto t = p->get_t();
            Matrix *weight = p->get_weight();
            Matrix *grad = p->get_grad();
            Matrix *mm = p->get_m();
            Matrix *mv = p->get_v();
            Shape shape = weight->getShape();
            grad->checkShape(shape);
            mm->checkShape(shape);
            mv->checkShape(shape);
            for (uint i = 0; i < shape.rowCnt; ++ i) {
                for (uint j = 0; j < shape.colCnt; ++ j) {
                    // 下面的都应该是device data
                    auto weight_data = weight->getLowLevelData();
                    auto grad_data = grad->getLowLevelData();
                    auto mm_data = mm->getLowLevelData();
                    auto mv_data = mv->getLowLevelData();
                    auto &value = weight_data[i*shape.colCnt + j];
                    auto &m = mm_data[i*shape.colCnt + j];
                    auto &v = mv_data[i*shape.colCnt + j];
                    auto &gradient = grad_data[i*shape.colCnt + j];
                    // auto &value = (*weight)[i][j];
                    // auto &m = (*mm)[i][j];
                    // auto &v = (*mv)[i][j];
                    // auto &gradient = (*grad)[i][j];
                    // fix me to device data
                    m = beta1 * m + (1 - beta1) * gradient;
                    v = beta2 * v + (1 - beta2) * std::pow(gradient, 2);
                    DATATYPE m_hat = m / (1 - std::pow(beta1, t));
                    DATATYPE v_hat = v / (1 - std::pow(beta2, t));
                    value -=  lr * (m_hat / (std::sqrt(v_hat) + epsilon));
                }
            }
        }

        cp_to_device();
    }

    void Adam::cuda_step() {
        for (auto p : parameters) {
            if (!p->require_grad()) {
                continue;
            }
            p->inc_t();
            auto t = p->get_t();
            g_gpu_backend_ops->step(lr, t, p->get_weight(), p->get_grad(), p->get_m(), p->get_v());
        }
    }

    void Adam::zero_grad() {
        for (auto p : parameters) {
            if (!p->require_grad()) {
                continue;
            }
            p->zero_grad();
        }
    }

    bool Adam::clip_grad(DATATYPE grad_clip_val) {
        cp_from_device();
        DATATYPE norm = 0;
        
        for (auto param : parameters) {
            if (!param->require_grad()) {
                continue;
            }
            
            auto grad = param->get_grad();
            auto grad_data = grad->getLowLevelData();
            Shape shape = grad->getShape();
            for (uint i = 0; i < shape.rowCnt; ++ i) {
                for (uint j = 0; j < shape.colCnt; ++ j) {
                    // norm += std::pow((*grad)[i][j], 2);
                    // fix me to device data
                    norm += std::pow(grad_data[i*shape.colCnt + j], 2);
                }
            }
        }
        norm = sqrt(norm);
        if (norm > grad_clip_val) {
            for (auto param : parameters) {
                if (!param->require_grad()) {
                    continue;
                }
                auto grad = param->get_grad();
                // fix me to device data
                *grad *= grad_clip_val / norm;
            }
        }
        cp_to_device();
        return norm > grad_clip_val;
    }

    bool Adam::cuda_clip_grad(DATATYPE grad_clip_val) {
        std::cerr << "Not implemented yet!" << std::endl;
        assert(false);
        return false;
    }

    void Adam::cp_from_device() {
        for (auto p : parameters) {
            p->get_weight()->cp_from_device();
            if (!p->require_grad()) {
                continue;
            }
            p->get_grad()->cp_from_device();
            p->get_m()->cp_from_device();
            p->get_v()->cp_from_device();
        }
    }

    void Adam::cp_to_device() {
        for (auto p : parameters) {
            p->get_weight()->cp_to_device();
            if (!p->require_grad()) {
                continue;
            }
            p->get_grad()->cp_to_device();
            p->get_m()->cp_to_device();
            p->get_v()->cp_to_device();
        }
    }

} // namespace autograd