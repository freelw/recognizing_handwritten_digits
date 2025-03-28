#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "autograd/node.h"
#include "autograd/parameters.h"

namespace transformer {
    class Adam {
        public:
            Adam(DATATYPE _lr,
                DATATYPE _beta1 = 0.9, DATATYPE _beta2 = 0.999, DATATYPE _epsilon = 1e-8)
                : lr(_lr), beta1(_beta1), beta2(_beta2), epsilon(_epsilon)
                {}
            void set_parameters(const std::vector<autograd::Parameters*> &_parameters);
            bool is_parameters_set();
            void step();
            void zero_grad();
            bool clip_grad(DATATYPE grad_clip_val);
        private:
            std::vector<autograd::Parameters*> parameters;
            DATATYPE lr; 
            DATATYPE beta1;
            DATATYPE beta2;
            DATATYPE epsilon;
    };
} // namespace transformer
#endif