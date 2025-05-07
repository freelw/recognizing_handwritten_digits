#ifndef COMMON_H
#define COMMON_H


#include "backends/cpu/cpu_ops.h"
#include "graph/node.h"

extern BackendOps *g_backend_ops;
void zero_grad();
void init_backend();
void release_backend();

#endif