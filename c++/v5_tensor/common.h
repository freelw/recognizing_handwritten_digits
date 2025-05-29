#ifndef COMMON_H
#define COMMON_H


#include "backends/cpu/cpu_ops.h"
#include "backends/gpu/cuda_ops.h"
#include "graph/node.h"

extern BackendOps *g_backend_ops;
extern bool g_training;
void zero_grad();
void zero_c_tensors();
void print_no_zero_tensor_names();
void insert_boundary_action();
void init_backend();
void release_backend();
void construct_env();
void destruct_env();
void use_gpu(bool use = true);
bool is_use_gpu();
void print_all_tensors();

#define NUM_STEPS 9
#define MAX_POSENCODING_LEN 10000
#define RESOURCE_NAME "../../resources/fra_preprocessed.txt"
#define SRC_VOCAB_NAME "../fra_vocab_builder/vocab_en.txt"
#define TGT_VOCAB_NAME "../fra_vocab_builder/vocab_fr.txt"
#define TEST_FILE "./test.txt"

#endif