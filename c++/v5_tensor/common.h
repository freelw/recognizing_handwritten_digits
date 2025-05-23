#ifndef COMMON_H
#define COMMON_H


#include "backends/cpu/cpu_ops.h"
#include "backends/gpu/cuda_ops.h"
#include "graph/node.h"

extern BackendOps *g_backend_ops;
extern bool g_training;
void zero_grad();
void insert_boundary_action();
void init_backend();
void release_backend();
void construct_env();
void destruct_env();
void use_gpu(bool use = true);
bool is_use_gpu();

#define NUM_STEPS 9
#define MAX_POSENCODING_LEN 10000

#endif