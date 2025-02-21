#include "mlp.h"

#include <cassert>
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>

Neuron::Neuron(int _inputSize, bool rand) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < _inputSize; i++) {
        if (rand) {
            weight.push_back(new Parameter(distribution(generator)));
        } else {
            weight.push_back(new Parameter(0.1));
        }
    }
    if (rand) {
        bias = new Parameter(distribution(generator));
    } else {
        bias = new Parameter(0.1);
    }
}

VariablePtr Neuron::forward(const std::vector<VariablePtr> &input) {
    assert(input.size() == weight.size());
    VariablePtr res = bias;
    for (int i = 0; i < weight.size(); i++) {
        res = *res + (*(weight[i])) * input[i];
    }
    return res->Relu();
}

void Neuron::update(double lr, int epoch) {
    // for (int i = 0; i < weight.size(); i++) {
    //     weight[i]->setValue(weight[i]->getValue() - lr * weight[i]->getGradient());
    // }
    // bias->setValue(bias->getValue() - lr * bias->getGradient());

    this->adamUpdate(lr, 0.9, 0.95, 1e-8, epoch);
}

void Neuron::adamUpdate(double lr, double beta1, double beta2, double epsilon, int epoch) {
    for (int i = 0; i < weight.size(); i++) {
        weight[i]->adamUpdate(lr, beta1, beta2, epsilon, epoch);
    }
    bias->adamUpdate(lr, beta1, beta2, epsilon, epoch);
}

void Neuron::zeroGrad() {
    for (int i = 0; i < weight.size(); i++) {
        weight[i]->zeroGrad();
    }
    bias->zeroGrad();
}

std::ostream & operator<<(std::ostream &output, const Neuron &s) {
    output << std::endl << "\t" << "weight : ";
    for (int i = 0; i < s.weight.size(); i++) {
        output << s.weight[i]->getValue() << " ";
    }
    output << s.bias->getValue() << std::endl;
    output << "\t" << "grad : ";
    for (int i = 0; i < s.weight.size(); i++) {
        output << s.weight[i]->getGradient() << " ";
    }
    output << s.bias->getGradient();
    return output;
}

Layer::Layer(int _inputSize, int _outputSize) : inputSize(_inputSize), outputSize(_outputSize) {
    
}

void Layer::zeroGrad() {
    
}

LinerLayer::LinerLayer(int _inputSize, int _outputSize, bool rand) : Layer(_inputSize, _outputSize) {
    for (int i = 0; i < _outputSize; i++) {
        neurons.push_back(new Neuron(_inputSize, rand));
    }
}

std::vector<VariablePtr> LinerLayer::forward(const std::vector<VariablePtr> &input) {
    assert(input.size() == inputSize);
    std::vector<VariablePtr> res;
    for (int i = 0; i < neurons.size(); i++) {
        res.push_back(neurons[i]->forward(input));
    }
    return res;
}

void LinerLayer::update(double lr, int epoch) {
    for (int i = 0; i < neurons.size(); i++) {
        neurons[i]->update(lr, epoch);
    }
}

std::ostream & operator<<(std::ostream &output, const LinerLayer &s) {
    output << "LinerLayer begin" << std::endl;
    for (int i = 0; i < s.neurons.size(); i++) {
        output << "neuron[" << i << "]: " << *(s.neurons[i]) << " " << std::endl;
    }
    output << "LinerLayer end" << std::endl;
    return output;
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

void ReluLayer::update(double lr, int epoch) {
    
}

Model::Model(int _inputSize, std::vector<int> _outputSizes, bool rand) {
    LinerLayer *linerLayer = new LinerLayer(_inputSize, _outputSizes[0], rand);
    layers.push_back(linerLayer);
    linerLayers.push_back(linerLayer);
    for (int i = 1; i < _outputSizes.size(); i++) {
        layers.push_back(new ReluLayer(_outputSizes[i - 1]));
        linerLayer = new LinerLayer(_outputSizes[i - 1], _outputSizes[i], rand);
        layers.push_back(linerLayer);
        linerLayers.push_back(linerLayer);
    }
    // std::cout << "linerLayers.size() : " << linerLayers.size() << std::endl;
}

std::vector<VariablePtr> Model::forward(const std::vector<VariablePtr> &input) {
    std::vector<VariablePtr> res = input;
    for (int i = 0; i < layers.size(); i++) {
        res = layers[i]->forward(res);
    }
    return res;
}

void Model::update(double lr, int epoch) {
    for (int i = 0; i < layers.size(); i++) {
        layers[i]->update(lr, epoch);
    }
}

void Model::zeroGrad() {
    for (int i = 0; i < layers.size(); i++) {
        layers[i]->zeroGrad();
    }
}

std::ostream & operator<<(std::ostream &output, const Model &s) {
    output << "Model begin" << std::endl;
    for (int i = 0; i < s.linerLayers.size(); i++) {
        output << i << " : " << std::endl << *(s.linerLayers[i]);
    }
    output << "Model end" << std::endl;
    return output;
}

VariablePtr CrossEntropyLoss(const std::vector<VariablePtr> &input, int target) {
    assert(target < input.size());
    auto sum = allocTmpVar(0);
    for (int i = 0; i < input.size(); i++) {
        sum = *sum + input[i]->exp();
    }
    
    return (*(*(input[target]->exp()) / sum)->log()) * allocTmpVar(-1);
}
