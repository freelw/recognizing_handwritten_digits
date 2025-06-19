#ifndef MLP_H
#define MLP_H

#include "variable.h"
#include <vector>
#include <ostream>
#include <random>
#include <chrono>

class Neuron {
public:
    Neuron(uint _inputSize, bool rand,
        std::normal_distribution<double>& d_w,
        std::normal_distribution<double>& d_b,
        std::default_random_engine& generator_w,
        std::default_random_engine& generator_b);
    VariablePtr forward(const std::vector<VariablePtr>& input);
    void update(double lr);
    void zeroGrad();
    friend std::ostream& operator<<(std::ostream& output, const Neuron& s);
private:
    std::vector<VariablePtr> weight;
    VariablePtr bias;
};

class Layer {
public:
    Layer(uint _inputSize, uint _outputSize);
    virtual std::vector<VariablePtr> forward(const std::vector<VariablePtr>& input) = 0;
    virtual void update(double lr) {};
    virtual void zeroGrad();
protected:
    uint inputSize;
    uint outputSize;
};

class LinerLayer : public Layer {
public:
    LinerLayer(uint _inputSize, uint _outputSize, bool rand);
    std::vector<VariablePtr> forward(const std::vector<VariablePtr>& input);
    virtual void update(double lr);
    virtual void zeroGrad();
private:
    std::vector<Neuron*> neurons;
};

class SigmoidLayer : public Layer {
public:
    SigmoidLayer(uint _inputSize);
    std::vector<VariablePtr> forward(const std::vector<VariablePtr>& input);
};

VariablePtr MSELoss(const std::vector<VariablePtr>& input, uint t);

class Model {
public:
    Model(uint _inputSize, std::vector<uint> _outputSizes, bool rand = true);
    virtual std::vector<VariablePtr> forward(const std::vector<VariablePtr>& input, bool train = true);
    virtual void update(double lr);
    virtual void zeroGrad();
private:
    std::vector<Layer*> layers;
    std::vector<LinerLayer*> linerLayers;
};

#endif