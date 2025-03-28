#include "dropout.h"

namespace autograd {
    Dropout::Dropout(DATATYPE _dropout) : dropout(_dropout) {
        assert(dropout > 0);
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        gen = std::mt19937(seed);
        dis = std::uniform_real_distribution<>(0, 1);
    }

    std::vector<Node *> Dropout::forward(const std::vector<Node *> &inputs) {
        std::vector<Node *> res;
        res.resize(inputs.size());
        for (uint j = 0; j < inputs.size(); j++) {
            auto &input = inputs[j];
            Matrix *mask = allocTmpMatrix(input->get_weight()->getShape());
            auto buffer = mask->getData();
            for (uint i = 0; i < mask->getShape().size(); i++) {
                buffer[i] = dis(gen) > dropout ? 1 : 0;
            }
            Node *n = allocNode(mask);
            res[j] = *input * n;
        }
        return res;
    }
} // namespace autograd