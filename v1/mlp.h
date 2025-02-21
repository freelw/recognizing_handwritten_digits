#ifndef MLP_H
#define MLP_H

#include "variable.h"
#include <vector>
#include <ostream>

class Neuron {
    public:
        Neuron(int _inputSize, bool rand);
        VariablePtr forward(const std::vector<VariablePtr> &input);
        void update(double lr);
        void zeroGrad();
        friend std::ostream & operator<<(std::ostream &output, const Neuron &s);
    private:
        std::vector<VariablePtr> weight;
        VariablePtr bias;
};

class Layer {
    public:
        Layer(int _inputSize, int _outputSize);
        virtual std::vector<VariablePtr> forward(const std::vector<VariablePtr> &input) = 0;
        virtual void update(double lr) = 0;
        virtual void zeroGrad();
    protected:
        int inputSize;
        int outputSize;
};

class LinerLayer : public Layer {
    public:
        LinerLayer(int _inputSize, int _outputSize, bool rand);
        std::vector<VariablePtr> forward(const std::vector<VariablePtr> &input);
        virtual void update(double lr);
        virtual void zeroGrad();
        friend std::ostream & operator<<(std::ostream &output, const LinerLayer &s);
    private:
        std::vector<Neuron*> neurons;
};

class ReluLayer : public Layer {
    public:
        ReluLayer(int _inputSize);
        std::vector<VariablePtr> forward(const std::vector<VariablePtr> &input);
        virtual void update(double lr);
};

VariablePtr CrossEntropyLoss(const std::vector<VariablePtr> &input, int target);

class Model {
    public:
        Model(int _inputSize, std::vector<int> _outputSizes, bool rand = true);
        std::vector<VariablePtr> forward(const std::vector<VariablePtr> &input);
        void update(double lr);
        void zeroGrad();
        friend std::ostream & operator<<(std::ostream &output, const Model &s);
    private:
        std::vector<Layer*> layers;
        std::vector<LinerLayer*> linerLayers;
};

#endif