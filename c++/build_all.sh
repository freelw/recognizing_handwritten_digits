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
    echo "v0 build failed"
    exit 1
fi
popd

pushd ./v2
make clean
make
if [ $? -ne 0 ]; then
    echo "v0 build failed"
    exit 1
fi
popd

pushd ./rnn
make clean
make
if [ $? -ne 0 ]; then
    echo "v0 build failed"
    exit 1
fi
popd
