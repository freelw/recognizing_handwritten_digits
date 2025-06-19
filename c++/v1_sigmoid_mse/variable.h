#ifndef VARIABLE_H
#define VARIABLE_H

#include <memory>
#include <ostream>
#include <vector>

class Variable;
typedef Variable* VariablePtr;

void destroyTmpVars();
void registerTmpVar(VariablePtr var);
VariablePtr allocTmpVar(double value);

class Variable {
public:
    Variable();
    Variable(double _value);
    Variable(double _value, double _gradient);
    virtual ~Variable() {}
    VariablePtr operator+(VariablePtr p);
    VariablePtr operator-(VariablePtr p);
    VariablePtr operator*(VariablePtr p);
    VariablePtr operator/(VariablePtr p);
    VariablePtr sqr();
    VariablePtr sigmoid();
    friend std::ostream& operator<<(std::ostream& output, const Variable& s);
    virtual void backward() = 0;
    double getValue() { return value; }
    double getGradient() { return gradient; }
    virtual void incGradient(double _gradient) { gradient += _gradient; }
    virtual void setGradient(double _gradient) { gradient = _gradient; }
    void setValue(double _value) { value = _value; }
    void bp();
    void incInputCount() { inputCount++; }
    void decInputCount() { inputCount--; }
    void zeroGrad();
    void adamUpdate(double lr, double beta1, double beta2, double epsilon, int t);
protected:
    double value;
    double gradient;
    VariablePtr parents[2];
    int inputCount;
    double m, v;
};

class TmpVar : public Variable {
public:
    TmpVar();
    TmpVar(double _value);
    TmpVar(double _value, double _gradient);
    void backward();
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

class MinusRes : public Variable {
public:
    MinusRes(VariablePtr _x, VariablePtr _y);
    void backward();
};

class MulRes : public Variable {
public:
    MulRes(VariablePtr _x, VariablePtr _y);
    void backward();
};

class DivRes : public Variable {
public:
    DivRes(VariablePtr _x, VariablePtr _y);
    void backward();
};

class SqrRes : public Variable {
public:
    SqrRes(VariablePtr _x);
    void backward();
};

class SigmoidRes : public Variable {
public:
    SigmoidRes(VariablePtr _x);
    void backward();
};
#endif