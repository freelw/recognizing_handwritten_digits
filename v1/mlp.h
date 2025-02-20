#ifndef MLP_H
#define MLP_H

#include "variable.h"
#include <vector>

class Neuron {
    public:
        Neuron(int _inputSize);
        VariablePtr forward(VariablePtr input);
        void update(double lr);
    private:
        std::vector<VariablePtr> weight;
        VariablePtr bias;
};

class Layer {
    public:
        Layer(int _inputSize, int _outputSize);
        virtual std::vector<VariablePtr> forward(const std::vector<VariablePtr> &input) = 0;
        virtual void update(double lr) = 0;
    protected:
        int inputSize;
        int outputSize;
};

class LinerLayer : public Layer {
    public:
        LinerLayer(int _inputSize, int _outputSize);
        std::vector<VariablePtr> forward(const std::vector<VariablePtr> &input);
        void update(double lr);
    private:
        std::vector<Neuron*> neurons;
};

class ReluLayer : public Layer {
    public:
        ReluLayer(int _inputSize);
        std::vector<VariablePtr> forward(const std::vector<VariablePtr> &input);
        void update(double lr);
};

VariablePtr CrossEntropyLoss(const std::vector<VariablePtr> &input, int target);

class Model {
    public:
        Model(int _inputSize, std::vector<int> _outputSizes);
        std::vector<VariablePtr> forward(const std::vector<VariablePtr> &input);
        void update(double lr);
    private:
        std::vector<Layer*> layers;
};

#endif