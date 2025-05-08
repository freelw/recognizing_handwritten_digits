#include "common.h"
#include "optimizers/parameter.h"

BackendOps *g_backend_ops = nullptr;
BackendOps *g_gpu_backend_ops = nullptr;

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
    g_backend_ops = new CPUOps();
    g_gpu_backend_ops = new CUDAOps();
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
    releaseTensorMem();
    release_backend();
}
