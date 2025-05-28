#include "common.h"
#include "optimizers/parameter.h"

BackendOps *g_backend_ops = nullptr;
bool g_training = true;

bool b_use_gpu = false;

void zero_grad() {
    gCreateAction(
        new ZeroGradAction()
    );
}

void zero_c_tensors() {
    gCreateAction(
        new ZeroCTensorsAction()
    );
}

void print_no_zero_tensor_names() {
    gCreateAction(
        new PrintNoZeroTensorNamesAction()
    );
}

void insert_boundary_action() {
    gCreateAction(
        new BoundaryAction()
    );
}

void init_backend() {
    if (b_use_gpu) {
        #ifndef GCC_ASAN
        g_backend_ops = new CUDAOps();
        #else
        std::cerr << "Warning: GPU backend is not available in ASAN build. Now use cpu instead!!!" << std::endl;
        g_backend_ops = new CPUOps();
        #endif
    } else {
        g_backend_ops = new CPUOps();
    }
}

void release_backend() {
    delete g_backend_ops;
    g_backend_ops = nullptr;
}

void construct_env() {
    init_backend();
}

void destruct_env() {
    // sanitizeTensors();
    releaseParameters();
    freeAllActions();
    freeAllTensors();
    freeAllCTensors();
    freeAllTensorViews();
    freeAllGradTensors();
    graph::freeAllNodes();
    graph::freeAllEdges();
    releaseTensorMem();
    release_backend();
}

void use_gpu(bool use) {
    b_use_gpu = use;
}

bool is_use_gpu() {
    return b_use_gpu;
}