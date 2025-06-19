#include "mlp.h"

#include <cassert>
#include <cmath>
#include <iostream>

Neuron::Neuron(
    uint _inputSize,
    std::normal_distribution<double>& d_w,
    std::normal_distribution<double>& d_b,
    std::default_random_engine& generator_w,
    std::default_random_engine& generator_b) {

    for (uint i = 0; i < _inputSize; i++) {
        weight.push_back(new Variable(d_w(generator_w)));
    }
    bias = new Variable(d_b(generator_b));
}

VariablePtr Neuron::forward(const std::vector<VariablePtr>& input) {
    assert(input.size() == weight.size());
    VariablePtr res = bias;
    for (uint i = 0; i < weight.size(); i++) {
        res = *res + (*(weight[i])) * input[i];
    }
    return res;
}

void Neuron::update(double lr) {
    for (uint i = 0; i < weight.size(); i++) {
        weight[i]->update(lr);
    }
    bias->update(lr);
}

void Neuron::zeroGrad() {
    for (uint i = 0; i < weight.size(); i++) {
        weight[i]->zeroGrad();
    }
    bias->zeroGrad();
}

Layer::Layer(uint _inputSize, uint _outputSize) : inputSize(_inputSize), outputSize(_outputSize) {

}

void Layer::zeroGrad() {

}

LinerLayer::LinerLayer(uint _inputSize, uint _outputSize) : Layer(_inputSize, _outputSize) {
    double stddev = sqrt(2. / (_inputSize + _outputSize)) * sqrt(2);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed);
    std::default_random_engine generator_b(seed + 1024);
    std::normal_distribution<double> distribution_w(0.0, stddev);
    std::normal_distribution<double> distribution_b(0.0, 0.02);
    for (uint i = 0; i < _outputSize; i++) {
        neurons.push_back(new Neuron(_inputSize, distribution_w, distribution_b, generator_w, generator_b));
    }
}

std::vector<VariablePtr> LinerLayer::forward(const std::vector<VariablePtr>& input) {
    assert(input.size() == inputSize);
    std::vector<VariablePtr> res;
    for (uint i = 0; i < neurons.size(); i++) {
        res.push_back(neurons[i]->forward(input));
    }
    return res;
}

void LinerLayer::update(double lr) {
    for (uint i = 0; i < neurons.size(); i++) {
        neurons[i]->update(lr);
    }
}

void LinerLayer::zeroGrad() {
    for (uint i = 0; i < neurons.size(); i++) {
        neurons[i]->zeroGrad();
    }
}

SigmoidLayer::SigmoidLayer(uint _inputSize) : Layer(_inputSize, _inputSize) {
    // No additional initialization needed for SigmoidLayer
}

std::vector<VariablePtr> SigmoidLayer::forward(const std::vector<VariablePtr>& input) {
    assert(input.size() == inputSize);
    std::vector<VariablePtr> res;
    for (uint i = 0; i < input.size(); i++) {
        res.push_back(input[i]->sigmoid());
    }
    return res;
}

Model::Model(uint _inputSize, std::vector<uint> _outputSizes) {
    LinerLayer* linerLayer = new LinerLayer(_inputSize, _outputSizes[0]);
    layers.push_back(linerLayer);
    for (uint i = 1; i < _outputSizes.size(); i++) {
        layers.push_back(new SigmoidLayer(_outputSizes[i - 1]));
        linerLayer = new LinerLayer(_outputSizes[i - 1], _outputSizes[i]);
        layers.push_back(linerLayer);
    }
}

std::vector<VariablePtr> Model::forward(const std::vector<VariablePtr>& input) {
    std::vector<VariablePtr> res = input;
    for (uint i = 0; i < layers.size(); i++) {
        res = layers[i]->forward(res);
    }
    return res;
}

void Model::update(double lr) {
    for (uint i = 0; i < layers.size(); i++) {
        layers[i]->update(lr);
    }
}

void Model::zeroGrad() {
    for (uint i = 0; i < layers.size(); i++) {
        layers[i]->zeroGrad();
    }
}

VariablePtr MSELoss(const std::vector<VariablePtr>& input, uint t) {
    assert(t >= 0);
    assert(t <= 9);
    std::vector<uint> target(10, 0);
    target[t] = 1;
    assert(target.size() == input.size());
    VariablePtr sum = allocTmpVar(0);
    for (uint i = 0; i < input.size(); i++) {
        VariablePtr diff = *input[i] - allocTmpVar(target[i]);
        sum = *sum + diff->sqr();
    }
    return *sum / allocTmpVar(input.size() * 2);
}
