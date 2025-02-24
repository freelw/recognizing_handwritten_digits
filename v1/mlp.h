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
            std::normal_distribution<double> & d_w,
            std::normal_distribution<double> & d_b,
            std::default_random_engine & generator_w,
            std::default_random_engine & generator_b);
        VariablePtr forward(const std::vector<VariablePtr> &input);
        void update(double lr, int epoch);
        void adamUpdate(double lr, double beta1, double beta2, double epsilon, int epoch);
        void zeroGrad();
        friend std::ostream & operator<<(std::ostream &output, const Neuron &s);
    private:
        std::vector<VariablePtr> weight;
        VariablePtr bias;
};

class Layer {
    public:
        Layer(uint _inputSize, uint _outputSize);
        virtual std::vector<VariablePtr> forward(const std::vector<VariablePtr> &input) = 0;
        virtual void update(double lr, int epoch) = 0;
        virtual void zeroGrad();
    protected:
        uint inputSize;
        uint outputSize;
};

class LinerLayer : public Layer {
    public:
        LinerLayer(uint _inputSize, uint _outputSize, bool rand);
        std::vector<VariablePtr> forward(const std::vector<VariablePtr> &input);
        virtual void update(double lr, int epoch);
        virtual void zeroGrad();
        friend std::ostream & operator<<(std::ostream &output, const LinerLayer &s);
    private:
        std::vector<Neuron*> neurons;
};

class ReluLayer : public Layer {
    public:
        ReluLayer(uint _inputSize);
        std::vector<VariablePtr> forward(const std::vector<VariablePtr> &input);
        virtual void update(double lr, int epoch);
};

VariablePtr CrossEntropyLoss(const std::vector<VariablePtr> &input, uint target);

class Model {
    public:
        Model(uint _inputSize, std::vector<uint> _outputSizes, bool rand = true);
        std::vector<VariablePtr> forward(const std::vector<VariablePtr> &input);
        void update(double lr, int epoch);
        void zeroGrad();
        friend std::ostream & operator<<(std::ostream &output, const Model &s);
    private:
        std::vector<Layer*> layers;
        std::vector<LinerLayer*> linerLayers;
};

#endif