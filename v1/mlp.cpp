#include "mlp.h"

#include <cassert>
#include <cmath>
#include <random>
#include <chrono>

Neuron::Neuron(int _inputSize) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < _inputSize; i++) {
        weight.push_back(new Parameter(distribution(generator)));
    }
    bias = new Parameter(distribution(generator));
}

VariablePtr Neuron::forward(VariablePtr input) {
    VariablePtr res = bias;
    for (int i = 0; i < weight.size(); i++) {
        res = *res + (*(weight[i])) * input;
    }
    return res->Relu();
}

void Neuron::update(double lr) {
    for (int i = 0; i < weight.size(); i++) {
        weight[i]->setGradient(weight[i]->getGradient() - lr * weight[i]->getGradient());
    }
    bias->setGradient(bias->getGradient() - lr * bias->getGradient());
}

void Neuron::zeroGrad() {
    for (int i = 0; i < weight.size(); i++) {
        weight[i]->zeroGrad();
    }
    bias->zeroGrad();
}

Layer::Layer(int _inputSize, int _outputSize) : inputSize(_inputSize), outputSize(_outputSize) {
    
}

void Layer::zeroGrad() {
    
}

LinerLayer::LinerLayer(int _inputSize, int _outputSize) : Layer(_inputSize, _outputSize) {
    for (int i = 0; i < _outputSize; i++) {
        neurons.push_back(new Neuron(_inputSize));
    }
}

std::vector<VariablePtr> LinerLayer::forward(const std::vector<VariablePtr> &input) {
    assert(input.size() == inputSize);
    std::vector<VariablePtr> res;
    for (int i = 0; i < neurons.size(); i++) {
        res.push_back(neurons[i]->forward(input[i]));
    }
    return res;
}

void LinerLayer::update(double lr) {
    for (int i = 0; i < neurons.size(); i++) {
        neurons[i]->update(lr);
    }
}

void LinerLayer::zeroGrad() {
    for (int i = 0; i < neurons.size(); i++) {
        neurons[i]->zeroGrad();
    }
}

ReluLayer::ReluLayer(int _inputSize) : Layer(_inputSize, _inputSize) {
    
}

std::vector<VariablePtr> ReluLayer::forward(const std::vector<VariablePtr> &input) {
    assert(input.size() == inputSize);
    std::vector<VariablePtr> res;
    for (int i = 0; i < input.size(); i++) {
        res.push_back(input[i]->Relu());
    }
    return res;
}

void ReluLayer::update(double lr) {
    
}

Model::Model(int _inputSize, std::vector<int> _outputSizes) {
    layers.push_back(new LinerLayer(_inputSize, _outputSizes[0]));
    for (int i = 1; i < _outputSizes.size(); i++) {
        layers.push_back(new ReluLayer(_outputSizes[i - 1]));
        layers.push_back(new LinerLayer(_outputSizes[i - 1], _outputSizes[i]));
    }
}

std::vector<VariablePtr> Model::forward(const std::vector<VariablePtr> &input) {
    std::vector<VariablePtr> res = input;
    for (int i = 0; i < layers.size(); i++) {
        res = layers[i]->forward(res);
    }
    return res;
}

void Model::update(double lr) {
    for (int i = 0; i < layers.size(); i++) {
        layers[i]->update(lr);
    }
}

void Model::zeroGrad() {
    for (int i = 0; i < layers.size(); i++) {
        layers[i]->zeroGrad();
    }
}

VariablePtr CrossEntropyLoss(const std::vector<VariablePtr> &input, int target) {
    assert(target < input.size());
    double sum = 0;
    for (int i = 0; i < input.size(); i++) {
        sum += std::exp(input[i]->getValue());
    }
    return allocTmpVar(-std::log(std::exp(input[target]->getValue()) / sum));
}
