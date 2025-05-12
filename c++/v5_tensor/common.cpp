#include "common.h"
#include "optimizers/parameter.h"

BackendOps *g_backend_ops = nullptr;

bool b_use_gpu = false;

void zero_grad() {
    gCreateAction(
        new ZeroGradAction()
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
        std::cerr << "Error: GPU backend is not available in ASAN build." << std::endl;
        abort();
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
    sanitizeTensors();
    releaseParameters();
    freeAllActions();
    freeAllTensors();
    freeAllTensorViews();
    freeAllGradTensors();
    graph::freeAllNodes();
    graph::freeAllEdges();
    graph::freeAllEmbs();
    releaseTensorMem();
    release_backend();
}

void use_gpu(bool use) {
    b_use_gpu = use;
}