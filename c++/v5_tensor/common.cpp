#include "common.h"

BackendOps *g_backend_ops = nullptr;

void zero_grad() {
    gCreateAction(
        new ZeroGradAction()
    );
}

void init_backend() {
    g_backend_ops = new CPUOps();
}

void release_backend() {
    
    delete g_backend_ops;
    g_backend_ops = nullptr;
}
