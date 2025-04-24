#!/bin/bash

NCU_PATH="/usr/local/cuda/nsight-compute-2024.3.2/ncu"
NSIGHT_OUT_PATH="/home/cs/project/recognizing_handwritten_digits/build/nsight_out/"
CMAKE_BUILD_PATH="/home/cs/project/recognizing_handwritten_digits/build/matrixmul/"
mkdir -p ${NSIGHT_OUT_PATH}
${NCU_PATH} --set detailed \
-o ${NSIGHT_OUT_PATH}/matrix_baseline \
${CMAKE_BUILD_PATH}/matrix_baseline

${NCU_PATH} --set detailed \
-o ${NSIGHT_OUT_PATH}/matrix_opt1 \
${CMAKE_BUILD_PATH}/matrix_opt1

${NCU_PATH} --set detailed \
-o ${NSIGHT_OUT_PATH}/matrix_opt2 \
${CMAKE_BUILD_PATH}/matrix_opt2