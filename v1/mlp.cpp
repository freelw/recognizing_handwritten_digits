#include "mlp.h"

#include <cassert>
#include <cmath>
#include <iostream>

Neuron::Neuron(
    uint _inputSize, bool rand,
    std::normal_distribution<double> & d_w,
    std::normal_distribution<double> & d_b,
    std::default_random_engine & generator_w,
    std::default_random_engine & generator_b) {

    for (uint i = 0; i < _inputSize; i++) {
        if (rand) {
            weight.push_back(new Parameter(d_w(generator_w)));
        } else {
            weight.push_back(new Parameter(0.1));
        }
    }

    if (rand) {
        bias = new Parameter(d_b(generator_b));
    } else {
        bias = new Parameter(0.1);
    }
}

VariablePtr Neuron::forward(const std::vector<VariablePtr> &input) {
    assert(input.size() == weight.size());
    VariablePtr res = bias;
    for (uint i = 0; i < weight.size(); i++) {
        res = *res + (*(weight[i])) * input[i];
    }
    return res;
}

void Neuron::update(double lr, int epoch) {
    this->adamUpdate(lr, 0.9, 0.95, 1e-8, epoch);
}

void Neuron::adamUpdate(double lr, double beta1, double beta2, double epsilon, int epoch) {
    for (uint i = 0; i < weight.size(); i++) {
        weight[i]->adamUpdate(lr, beta1, beta2, epsilon, epoch);
    }
    bias->adamUpdate(lr, beta1, beta2, epsilon, epoch);
}

void Neuron::zeroGrad() {
    for (uint i = 0; i < weight.size(); i++) {
        weight[i]->zeroGrad();
    }
    bias->zeroGrad();
}

std::ostream & operator<<(std::ostream &output, const Neuron &s) {
    output << std::endl << "\t" << "weight : ";
    for (uint i = 0; i < s.weight.size(); i++) {
        output << s.weight[i]->getValue() << " ";
    }
    output << s.bias->getValue() << std::endl;
    output << "\t" << "grad : ";
    for (uint i = 0; i < s.weight.size(); i++) {
        output << s.weight[i]->getGradient() << " ";
    }
    output << s.bias->getGradient();
    return output;
}

Layer::Layer(uint _inputSize, uint _outputSize) : inputSize(_inputSize), outputSize(_outputSize) {
    
}

void Layer::zeroGrad() {
    
}

LinerLayer::LinerLayer(uint _inputSize, uint _outputSize, bool rand) : Layer(_inputSize, _outputSize) {
    double stddev = sqrt(2./(_inputSize + _outputSize))*sqrt(2);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed);
    std::default_random_engine generator_b(seed+1024);
    std::normal_distribution<double> distribution_w(0.0, stddev);
    std::normal_distribution<double> distribution_b(0.0, 0.02);
    for (uint i = 0; i < _outputSize; i++) {
        neurons.push_back(new Neuron(_inputSize, rand, distribution_w, distribution_b, generator_w, generator_b));
    }
}

std::vector<VariablePtr> LinerLayer::forward(const std::vector<VariablePtr> &input) {
    assert(input.size() == inputSize);
    std::vector<VariablePtr> res;
    for (uint i = 0; i < neurons.size(); i++) {
        res.push_back(neurons[i]->forward(input));
    }
    return res;
}

void LinerLayer::update(double lr, int epoch) {
    for (uint i = 0; i < neurons.size(); i++) {
        neurons[i]->update(lr, epoch);
    }
}

std::ostream & operator<<(std::ostream &output, const LinerLayer &s) {
    output << "LinerLayer begin" << std::endl;
    for (uint i = 0; i < s.neurons.size(); i++) {
        output << "neuron[" << i << "]: " << *(s.neurons[i]) << " " << std::endl;
    }
    output << "LinerLayer end" << std::endl;
    return output;
}

void LinerLayer::zeroGrad() {
    for (uint i = 0; i < neurons.size(); i++) {
        neurons[i]->zeroGrad();
    }
}

ReluLayer::ReluLayer(uint _inputSize) : Layer(_inputSize, _inputSize) {
    
}

std::vector<VariablePtr> ReluLayer::forward(const std::vector<VariablePtr> &input) {
    assert(input.size() == inputSize);
    std::vector<VariablePtr> res;
    for (uint i = 0; i < input.size(); i++) {
        res.push_back(input[i]->Relu());
    }
    return res;
}

void ReluLayer::update(double lr, int epoch) {
    
}

DropoutLayer::DropoutLayer(uint _inputSize, double _p) : Layer(_inputSize, _inputSize), p(_p) {
    
}

std::vector<VariablePtr> DropoutLayer::forward(const std::vector<VariablePtr> &input) {
    assert(input.size() == inputSize);
    std::vector<VariablePtr> res;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (uint i = 0; i < input.size(); i++) {
        if (dis(gen) < p) {
            res.push_back(allocTmpVar(0));
        } else {
            res.push_back(*(input[i])/allocTmpVar(1-p));
        }
    }
    return res;
}

void DropoutLayer::update(double lr, int epoch) {
    
}

Model::Model(uint _inputSize, std::vector<uint> _outputSizes, bool rand) {
    LinerLayer *linerLayer = new LinerLayer(_inputSize, _outputSizes[0], rand);
    layers.push_back(linerLayer);
    linerLayers.push_back(linerLayer);
    for (uint i = 1; i < _outputSizes.size(); i++) {
        layers.push_back(new ReluLayer(_outputSizes[i - 1]));
        linerLayer = new LinerLayer(_outputSizes[i - 1], _outputSizes[i], rand);
        layers.push_back(linerLayer);
        linerLayers.push_back(linerLayer);
    }
}

std::vector<VariablePtr> Model::forward(const std::vector<VariablePtr> &input, bool) {
    std::vector<VariablePtr> res = input;
    for (uint i = 0; i < layers.size(); i++) {
        res = layers[i]->forward(res);
    }
    return res;
}

void Model::update(double lr, int epoch) {
    for (uint i = 0; i < layers.size(); i++) {
        layers[i]->update(lr, epoch);
    }
}

void Model::zeroGrad() {
    for (uint i = 0; i < layers.size(); i++) {
        layers[i]->zeroGrad();
    }
}

std::ostream & operator<<(std::ostream &output, const Model &s) {
    output << "Model begin" << std::endl;
    for (uint i = 0; i < s.linerLayers.size(); i++) {
        output << i << " : " << std::endl << *(s.linerLayers[i]);
    }
    output << "Model end" << std::endl;
    return output;
}

ModelWithDropout::ModelWithDropout(uint _inputSize, std::vector<uint> _outputSizes, double p, bool rand) {
    LinerLayer *linerLayer = new LinerLayer(_inputSize, _outputSizes[0], rand);
    DropoutLayer *dropoutLayer = new DropoutLayer(_outputSizes[0], p);
    layers.push_back(linerLayer);
    layers.push_back(dropoutLayer);
    for (uint i = 1; i < _outputSizes.size(); i++) {
        layers.push_back(new ReluLayer(_outputSizes[i - 1]));
        linerLayer = new LinerLayer(_outputSizes[i - 1], _outputSizes[i], rand);
        dropoutLayer = new DropoutLayer(_outputSizes[i], p);
        layers.push_back(linerLayer);
    }
}

std::vector<VariablePtr> ModelWithDropout::forward(const std::vector<VariablePtr> &input, bool train) {
    std::vector<VariablePtr> res = input;
    for (uint i = 0; i < layers.size(); i++) {
        if (layers[i]->isDropout()) {
            if (train) {
                res = layers[i]->forward(res);
            }
        } else {
            res = layers[i]->forward(res);
        }
    }
    return res;
}

void ModelWithDropout::update(double lr, int epoch) {
    for (uint i = 0; i < layers.size(); i++) {
        layers[i]->update(lr, epoch);
    }
}

void ModelWithDropout::zeroGrad() {
    for (uint i = 0; i < layers.size(); i++) {
        layers[i]->zeroGrad();
    }
}

VariablePtr CrossEntropyLoss(const std::vector<VariablePtr> &input, uint target) {
    assert(target < input.size());
    double max_value = input[0]->getValue();
    for (uint i = 1; i < input.size(); i++) {
        if (input[i]->getValue() > max_value) {
            max_value = input[i]->getValue();
        }
    }
    auto sum = allocTmpVar(0);
    for (uint i = 0; i < input.size(); i++) {
        sum = *sum + (*input[i]+allocTmpVar(-max_value))->exp();
    }
    
    return (*(*((*input[target]+allocTmpVar(-max_value))->exp()) / sum)->log()) * allocTmpVar(-1);
}
