#ifndef COMMON_H
#define COMMON_H


#include "backends/cpu/cpu_ops.h"
#include "backends/gpu/cuda_ops.h"
#include "graph/node.h"

extern BackendOps *g_backend_ops;
void zero_grad();
void insert_boundary_action();
void init_backend();
void release_backend();
void construct_env();
void destruct_env();
void use_gpu(bool use = true);
bool is_use_gpu();

#endif