#!/bin/bash
export RELEASE=1

pushd ./transformer
make clean
make
if [ $? -ne 0 ]; then
    echo "transformer build failed"
    exit 1
fi
popd

pushd ./seq2seq
make clean
make
if [ $? -ne 0 ]; then
    echo "seq2seq build failed"
    exit 1
fi
popd

pushd ./deep_gru
make clean
make
if [ $? -ne 0 ]; then
    echo "deep_gru build failed"
    exit 1
fi
popd

pushd ./gru_embedding
make clean
make
if [ $? -ne 0 ]; then
    echo "gru_embedding build failed"
    exit 1
fi
popd

pushd ./gru
make clean
make
if [ $? -ne 0 ]; then
    echo "gru build failed"
    exit 1
fi
popd

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

pushd ./rnn_v1
make clean
make
if [ $? -ne 0 ]; then
    echo "rnn_v1 build failed"
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


pushd ./lstm_v1
make clean
make
if [ $? -ne 0 ]; then
    echo "lstm_v1 build failed"
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