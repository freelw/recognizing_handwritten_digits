#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "layers/layers.h"
class Adam {
    public:
        Adam(std::vector<Parameters*> _parameters, float _lr,
            float _beta1 = 0.9, float _beta2 = 0.95, float _epsilon = 1e-8)
            : parameters(_parameters), lr(_lr),
                beta1(_beta1), beta2(_beta2), epsilon(_epsilon)
            {}
        void step();
    private:
        std::vector<Parameters*> parameters;
        float lr; 
        float beta1;
        float beta2;
        float epsilon;
};
#endif