#ifndef VARIABLE_H
#define VARIABLE_H

#include <memory>
#include <ostream>

class Variable;
typedef std::shared_ptr<Variable> VariablePtr;
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
    private:
        double value;
        double gradient;
};
#endif