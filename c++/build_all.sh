#!/bin/bash

pushd ./v0
make clean
make
if [ $? -ne 0 ]; then
    echo "v0 build failed"
    exit 1
fi
popd

pushd ./v1
make clean
make
if [ $? -ne 0 ]; then
    echo "v1 build failed"
    exit 1
fi
popd

pushd ./v2
make clean
make
if [ $? -ne 0 ]; then
    echo "v2 build failed"
    exit 1
fi
popd

pushd ./v3
make clean
make
if [ $? -ne 0 ]; then
    echo "v3 build failed"
    exit 1
fi
popd

pushd ./rnn
make clean
make
if [ $? -ne 0 ]; then
    echo "rnn build failed"
    exit 1
fi
popd

pushd ./lstm
make clean
make
if [ $? -ne 0 ]; then
    echo "lstm build failed"
    exit 1
fi
popd

pushd ./matrix_bench
make clean
make
if [ $? -ne 0 ]; then
    echo "matrix_bench build failed"
    exit 1
fi
popd