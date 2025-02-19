#ifndef VARIABLE_H
#define VARIABLE_H

#include <memory>
#include <ostream>
#include <vector>

class Variable;
typedef Variable* VariablePtr;

void destroyTmpVars();
void registerTmpVar(VariablePtr var);

class Variable {
    public:
        Variable();
        Variable(double _value);
        Variable(double _value, double _gradient);
        VariablePtr operator+(VariablePtr p);
        VariablePtr operator*(VariablePtr p);
        VariablePtr Relu();
        VariablePtr log();
        VariablePtr exp();
        friend std::ostream & operator<<(std::ostream &output, const Variable &s);
        virtual void backward() = 0;
        double getValue() { return value; }
        double getGradient() { return gradient; }
        void incGradient(double _gradient) { gradient += _gradient; }
        void setGradient(double _gradient) { gradient = _gradient; }
    protected:
        double value;
        double gradient;
        std::vector<VariablePtr> parents;
};

class Parameter : public Variable {
    public:
        Parameter();
        Parameter(double _value);
        Parameter(double _value, double _gradient);
        void backward();
};

class AddRes : public Variable {
    public:
        AddRes(VariablePtr _x, VariablePtr _y);
        void backward();
};

class MulRes : public Variable {
    public:
        MulRes(VariablePtr _x, VariablePtr _y);
        void backward();
};

class ReluRes : public Variable {
    public:
        ReluRes(VariablePtr _x);
        void backward();
};

class LogRes : public Variable {
    public:
        LogRes(VariablePtr _x);
        void backward();
};

class ExpRes : public Variable {
    public:
        ExpRes(VariablePtr _x);
        void backward();
};


#endif