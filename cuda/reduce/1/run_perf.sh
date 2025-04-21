#!/bin/bash

NCU_PATH="/usr/local/cuda/nsight-compute-2024.3.2/ncu"
NSIGHT_OUT_PATH="/home/cs/project/recognizing_handwritten_digits/build/nsight_out/"
CMAKE_BUILD_PATH="/home/cs/project/recognizing_handwritten_digits/build/1/"
mkdir -p ${NSIGHT_OUT_PATH}
${NCU_PATH} --set detailed \
-o ${NSIGHT_OUT_PATH}/baseline \
${CMAKE_BUILD_PATH}/baseline

${NCU_PATH} --set detailed \
-o ${NSIGHT_OUT_PATH}/opt1 \
${CMAKE_BUILD_PATH}/opt1

${NCU_PATH} --set detailed \
-o ${NSIGHT_OUT_PATH}/opt2 \
${CMAKE_BUILD_PATH}/opt2

${NCU_PATH} --set detailed \
-o ${NSIGHT_OUT_PATH}/opt3 \
${CMAKE_BUILD_PATH}/opt3

${NCU_PATH} --set detailed \
-o ${NSIGHT_OUT_PATH}/opt4 \
${CMAKE_BUILD_PATH}/opt4

${NCU_PATH} --set detailed \
-o ${NSIGHT_OUT_PATH}/opt5 \
${CMAKE_BUILD_PATH}/opt5

${NCU_PATH} --set detailed \
-o ${NSIGHT_OUT_PATH}/opt6 \
${CMAKE_BUILD_PATH}/opt6

${NCU_PATH} --set detailed \
-o ${NSIGHT_OUT_PATH}/opt7 \
${CMAKE_BUILD_PATH}/opt7
